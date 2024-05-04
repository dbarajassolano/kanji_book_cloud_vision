from google.cloud import vision
from dataclasses import dataclass
from typing import List
from PIL import Image, ImageDraw, ImageFont
import io
import re
import tomli
import yaml
import argparse

@dataclass
class Word:
    string: str
    yb: int
    yt: int
    xl: int
    xr: int
    iskanji: bool
    
    def copy(self) -> "Word":
        newword: Word = Word.__new__(Word)
        newword.__dict__ = self.__dict__.copy()
        return newword

    def merge(self, next: "Word"):
        self.string = self.string + next.string
        self.xr     = next.xr

    def to_yaml_dict(self) -> dict:
        if self.iskanji:
            out = {'string': self.string, 'type': 'kanji'}
        else:
            string_encoded = self.string.encode('utf-8').decode('utf-8')
            out = {'string': string_encoded, 'type': 'vocab'}
        return out

    def __str__(self):
        return self.string

def read_config(file: str) -> dict:
    """Read the config TOML file"""
    
    with open(file, 'rb') as f:
        config_dict = tomli.load(f)

    return config_dict

def cloud_vision_read(file: str) -> vision.AnnotateFileResponse:
    """Read an image file and call Cloud Vision."""
    
    with io.open(file, 'rb') as image_file:
        content = image_file.read()

    client   = vision.ImageAnnotatorClient()
    image    = vision.Image(content=content)
    features = [{'type_': vision.Feature.Type.DOCUMENT_TEXT_DETECTION}]
    context  = {
        'language_hints': ['ja', 'en', 'ko'],
        'text_detection_params': vision.TextDetectionParams(
            enable_text_detection_confidence_score=True
        )
    }
    response = client.annotate_image(
        {
            'image': image,
            'features': features,
            'image_context': context
        }
    )

    return response

def words_raw_from_response(response: vision.AnnotateFileResponse, config: dict) -> List[Word]:
    """Get list of raw words from the Cloud Vision response"""
    
    words_raw = []
    for page in response.full_text_annotation.pages:
        for block in page.blocks:
            for paragraph in block.paragraphs:
                for word in paragraph.words:
                    word_text = ''.join([symbol.text for symbol in word.symbols])
                    bbox = word.bounding_box
                    bheight = abs(bbox.vertices[2].y - bbox.vertices[0].y)
                    iskanji = False
                    if word.confidence >= config['min_confidence'] and bheight >= config['min_word_height']:
                        if bheight >= config['min_kanji_height']:
                            iskanji = True
                        words_raw.append(
                            Word(string=word_text,
                                 yb=bbox.vertices[2].y,
                                 yt=bbox.vertices[0].y,
                                 xl=bbox.vertices[0].x,
                                 xr=bbox.vertices[1].x,
                                 iskanji=iskanji
                                )
                        )
    return words_raw

def merge_and_filter_words(words_raw: List[Word], config: dict) -> list:
    """Merge raw words into single words if they're sufficiently close to
    each other, filter out alphabetical strings, and group words by
    kanji.
    """
    regex = re.compile(r'[A-Za-z0-9/-]')
    words_raw = [w for w in words_raw if not regex.match(w.string)]

    words = []
    buffer_set = False
    buffer = words_raw[0].copy()
    for idx in range(len(words_raw) - 1):
        # Word is a kanji, so append and skip
        if words_raw[idx].iskanji == True:
            words.append(words_raw[idx].copy())
            buffer_set = False
            continue
        if buffer_set == False:
            buffer = words_raw[idx].copy()
        if abs(words_raw[idx + 1].xl - words_raw[idx].xr) <= config['min_interword_distance']:
            buffer.merge(words_raw[idx + 1])
            buffer_set = True
        else:
            words.append(buffer)
            buffer_set = False
    if buffer_set == True:
        words.append(buffer)
    else:
        words.append(words_raw[-1])

    # Sort words by kanji
    queries = []
    for word in words:
        if word.iskanji == True:
            queries.append(word)
            kanjit = word.yt - config['max_distance_from_kanji_top']
            kanjib = word.yb + config['max_distance_from_kanji_bottom']
            for k in words:
                if (k.yt >= kanjit) and (k.yb <= kanjib) and k.iskanji == False:
                    queries.append(k)

    return queries

def display(img_file: str, words: List[Word], conf: dict) -> None:

    img = Image.open(img_file).convert('RGB')
    # Draw the queries
    draw = ImageDraw.Draw(img)
    color = 'red'
    anchor = 'lb'
    font = ImageFont.truetype('Hiragino Sans GB.ttc', conf['display_font_size'])
    for word in words:
        draw.rectangle(((word.xl, word.yt), (word.xr, word.yb)), outline='blue')
        if word.iskanji:
            draw.text((word.xl, word.yt), word.string, color, font=font, anchor=anchor)
        else:
            draw.text((word.xl, word.yt), word.string, color, font=font, anchor=anchor)
    img.show()

def main() -> None:
    
    parser = argparse.ArgumentParser(description='Read kanjis and japanese vocabulary words from an image using Google Cloud Vision')
    parser.add_argument('img_file', metavar='I', type=str, help='Input image')
    parser.add_argument('out_file', metavar='O', type=str, help='Output YAML file')
    parser.add_argument('--conf',   metavar='C', type=str, help='Configuration file', default='conf.toml')
    args = parser.parse_args()
    img_file  = args.img_file
    out_file  = args.out_file
    conf_file = args.conf
    
    config = read_config(conf_file)
    response = cloud_vision_read(img_file)
    words_raw = words_raw_from_response(response, config)
    queries = merge_and_filter_words(words_raw, config)

    with open(out_file, 'w', encoding='utf-8') as output_file:
        dump = yaml.dump([q.to_yaml_dict() for q in queries],
                         allow_unicode=True)
        output_file.write(dump)

    display(img_file, queries, config)

if __name__ == '__main__':
    main()

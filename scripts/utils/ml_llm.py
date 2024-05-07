import re
import pathlib

from jinja2 import Environment, FileSystemLoader, select_autoescape


class CustomFilters(object):

  @staticmethod
  def norm_description(s: str) -> str:
    # remove emojis
    s = s.encode('ascii', 'ignore').decode('ascii')

    # remove non-ascii characters
    s = re.sub(r'[^\x00-\x7F]+', '', s)

    # convert newlines to full stops
    s = s.replace('\n', '. ')

    # remove multiple spaces
    s = re.sub(r'\s+', ' ', s)

    # remove space before punctuation
    s = re.sub(r'\s([?.!:"](?:\s|$))', r'\1', s)

    # remove double punctuation
    s = re.sub(r'([?.!"])([?.!"])+', r'\1', s)

    return s

  @staticmethod
  def norm_scalar(s: float | int,
                  suffix: str = '',
                  default_value: str = 'tidak disebutkan') -> str:
    if s == 0:
      return default_value

    return f"{s}{suffix}"

  @staticmethod
  def num_max(x, y):
    if x > y:
      return x
    return y

  @staticmethod
  def translate_compass(direction: str):
    if direction == "Utara":
      return "North"
    elif direction == "Selatan":
      return "South"
    elif direction == "Barat":
      return "West"
    elif direction == "Timur":
      return "East"
    elif direction == "Barat Daya":
      return "Northeast"
    elif direction == "Barat Laut":
      return "Northeest"
    elif direction == "Timur Laut":
      return "Southeest"
    elif direction == "Tenggara":
      return "Southeast"
    else:
      return direction

  @staticmethod
  def translate_gaya(s: str):
    return s.replace("Minimalis", "Minimalist")

  @staticmethod
  def translate_pemandangan(s: str):
    if s == 'Pemukiman Warga':
      return 'neighborhood'
    elif s == 'Pedesaan':
      return 'village'
    elif s == 'Perkotaan':
      return 'cities'
    elif s == 'Pegunungan':
      return 'mountains'
    elif s == 'Taman Kota':
      return 'park'
    else:
      return s

  @staticmethod
  def translate_water_source(s: str):
    if s == 'Sumur Bor':
      return 'wells'
    elif s == 'PAM atau PDAM':
      return 'municipal water'
    elif s == 'Sumur Resapan':
      return 'wells'
    elif s == 'Sumur Pompa':
      return 'wells'
    else:
      return s


class EmbeddingDocumentTemplateEngine(object):

  def __init__(self, template_name: str) -> None:
    # find template path
    template_path = pathlib.Path(__file__) \
        .parent \
        .parent \
        .resolve() \
        .joinpath('templates')

    # create Jinja environment
    fs_loader = FileSystemLoader(template_path)
    self.env = Environment(loader=fs_loader, autoescape=select_autoescape())

    # add custom filters
    self.env.filters['norm_description'] = CustomFilters.norm_description
    self.env.filters['norm_scalar'] = CustomFilters.norm_scalar
    self.env.filters['num_max'] = CustomFilters.num_max
    self.env.filters['translate_compass'] = CustomFilters.translate_compass
    self.env.filters['translate_gaya'] = CustomFilters.translate_gaya
    self.env.filters[
        'translate_pemandangan'] = CustomFilters.translate_pemandangan
    self.env.filters[
        'translate_water_source'] = CustomFilters.translate_water_source

    # load template
    self.template = self.env.get_template(template_name)

  def __call__(self, tp: dict) -> str:
    return self.template.render(**tp)

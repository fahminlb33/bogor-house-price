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

        # load template
        self.template = self.env.get_template(template_name)

    def __call__(self, tp: dict) -> str:
        return self.template.render(**tp)

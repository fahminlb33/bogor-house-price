import re
import pathlib

from jinja2 import Environment, FileSystemLoader, select_autoescape


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
        self.env.filters[
            'norm_description'] = EmbeddingDocumentTemplateEngine.norm_description
        self.env.filters[
            'norm_facilities'] = EmbeddingDocumentTemplateEngine.norm_facilities
        self.env.filters[
            'norm_house_mat'] = EmbeddingDocumentTemplateEngine.norm_house_mat
        self.env.filters['norm_tag'] = EmbeddingDocumentTemplateEngine.norm_tag
        self.env.filters[
            'norm_scalar'] = EmbeddingDocumentTemplateEngine.norm_scalar
        self.env.filters['num_max'] = EmbeddingDocumentTemplateEngine.num_max

        # load template
        self.template = self.env.get_template(template_name)

    def __call__(self, tp: dict) -> str:
        return self.template.render(row=tp)

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
    def norm_facilities(tp) -> str:
        s = ""

        if tp.facility_ac > 0:
            s += "AC, "
        if tp.facility_keamanan > 0:
            s += "keamanan/satpam, "
        if tp.facility_laundry > 0:
            s += "laundry, "
        if tp.facility_masjid > 0:
            s += "masjid, "
        if tp.ruang_makan > 0:
            s += "ruang makan, "
        if tp.ruang_tamu > 0:
            s += "ruang tamu, "

        if s == "":
            return "tidak disebutkan"

        return s[:-2]

    @staticmethod
    def norm_house_mat(tp) -> str:
        s = ""

        if tp.house_mat_bata_hebel > 0:
            s += "bata hebel, "
        if tp.house_mat_bata_merah > 0:
            s += "bata merah, "

        if s == "":
            return "tidak disebutkan"

        return s[:-2]

    @staticmethod
    def norm_tag(tp) -> str:
        s = ""

        if tp.tag_cash_bertahap > 0:
            s += "cash bertahap, "
        if tp.tag_komplek > 0:
            s += "komplek, "
        if tp.tag_kpr > 0:
            s += "KPR, "
        if tp.tag_perumahan > 0:
            s += "perumahan, "

        if s == "":
            return "tidak disebutkan"

        return s[:-2]

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

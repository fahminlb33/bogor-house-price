from flask import Blueprint, render_template

from utils.shared import cache

router = Blueprint('about', __name__)


@router.route('/about')
@cache.cached()
def page():
  return render_template(f'pages/about.html')

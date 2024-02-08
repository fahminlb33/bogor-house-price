from flask import (Blueprint, flash, g, redirect, render_template, request,
                   session, url_for)

router = Blueprint('about', __name__, url_prefix='/about')


@router.route("/")
def index():
    return render_template("about.html")

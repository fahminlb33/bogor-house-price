from flask import (Blueprint, flash, g, redirect, render_template, request,
                   session, url_for)

router = Blueprint('ask_ai', __name__, url_prefix='/ask-ai')


@router.route("/")
def index():
    return render_template("ask_ai.html")

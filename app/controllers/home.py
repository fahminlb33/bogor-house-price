from flask import (Blueprint, flash, g, redirect, render_template, request,
                   session, url_for)

router = Blueprint('home', __name__, url_prefix='/')


@router.route("/")
def index():
    statistic_cards = [
        ("Rumah", "20.012"),
        ("Rumah", "20.012"),
        ("Rumah", "20.012"),
        ("Rumah", "20.012"),
    ]

    return render_template("home.html", stats_cards=statistic_cards)


@router.route("/data/table")
def data_table():
    return {
        "data": [
            {
                "id": 1,
                "name": "John Doe",
                "age": 20
            },
            {
                "id": 2,
                "name": "Jane Doe",
                "age": 22
            },
            {
                "id": 3,
                "name": "John Smith",
                "age": 25
            },
            {
                "id": 4,
                "name": "Jane Smith",
                "age": 27
            },
        ]
    }

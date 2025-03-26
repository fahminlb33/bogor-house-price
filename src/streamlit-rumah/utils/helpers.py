from babel.numbers import format_compact_currency, format_currency


def formatter_pvalue(x):
    return "background-color: red" if x < 0.05 else None


def format_price(x):
    return format_compact_currency(x, "IDR", locale="id_ID")


def format_price_long(x):
    return format_currency(x, "IDR", locale="id_ID")


def percent_change(x, y):
    return (x - y) / y

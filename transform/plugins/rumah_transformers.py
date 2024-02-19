import re

from rumah_constants import (FACILITY_NAME_OUTLIERS, FACILITY_NAME_RULESET,
                             COMPANY_NAME_COMMONS, COMPANY_NAME_STOPWORDS,
                             COMPANY_NAME_INDEPENDENT)


def mask_name(s: str) -> str:
    if s is None or len(s) < 1:
        return None

    combined = ""
    parts = s.split()
    for part in parts:
        combined += part[:1] + "x" * (len(part) - 1) + " "

    return combined.strip()


def mask_phone(s: str) -> str:
    if s is None or len(s) < 3:
        return None

    return ("x" * (len(s) - 3)) + s[-3:]


def clean_facility(name: str) -> str:
    # set uppercase
    name = name.upper().strip()
    if len(name) < 1:
        return None

    # remove outlier
    for outlier in FACILITY_NAME_OUTLIERS:
        if outlier in name:
            return None

    # return rule if found
    for rule in FACILITY_NAME_RULESET:
        if rule in name:
            name = FACILITY_NAME_RULESET[rule]
            break

    return name.replace(" ", "_").lower()


def clean_agency_company(name: str) -> str:
    # return empty if null
    if name is None:
        return None

    # upper and strip
    name = name.upper().strip()

    # special case
    if "CENTURY 21" in name or "CENTURY21" in name:
        return "CENTURY21"

    # normalize common words
    if "PROPERTY" in name:
        name = name.replace("PROPERTY", "PROPERTI")

    # return common property names
    for company in COMPANY_NAME_COMMONS:
        if company in name:
            return company

    # remove common city names
    for word in COMPANY_NAME_STOPWORDS:
        if word in name:
            name = name.replace(word, "")

    # remove double spaces
    name = re.sub(' +', ' ', name).strip()

    # empty is independent
    if len(name) == 0:
        name = "INDEPENDENT"

    # check if this is independent
    for cn in COMPANY_NAME_INDEPENDENT:
        if cn in name:
            return "INDEPENDENT"

    # if the final name is just "PROPERTI", return independent
    if name == "PROPERTI":
        return "INDEPENDENT"

    return name

QUERY_OKRA_DATA_PG = """
    SELECT
        id,
        date,
        title,
        text,
        url,
        stars,
        raw
    FROM
        okra_okrareviews
    WHERE
        date >= '{date_from}'
    AND
        date <= '{date_to}';
"""

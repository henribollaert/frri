def bold(data, optimum='max', format_string="%.3f"):
    """
    Returns a pandas dataframe with formatted strings and bolded maximaL values.
    :param data:
    :param optimum:
    :param format_string:

    :return: data with bolded specified optimum
    """
    if optimum == 'max':
        optima = data != data.max()
    else:
        optima = data != data.min()
    bolded = data.apply(lambda x: "\\textbf{%s}" % format_string % x)
    formatted = data.apply(lambda x: format_string % x)
    return formatted.where(optima, bolded)

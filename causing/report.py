# -*- coding: utf-8 -*-
"""Create reportlab report."""

# pylint: disable=invalid-name
# spyder cannot read good-names from .pylintrc

from copy import copy, deepcopy

from numpy import array, empty, median, zeros
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from reportlab.platypus.flowables import KeepTogether

from causing import utils

# reportlab setting
#   10: standard text fontsize
#   12: Heading2 fontsize
#   20: Title fontsize
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='JustifyAlign',
                          parent=ParagraphStyle('Normal'),
                          alignment=TA_JUSTIFY))
styles.add(ParagraphStyle(name='TitleHelvetica',
                          parent=ParagraphStyle('Normal'),
                          fontName='Helvetica-Bold',
                          fontSize=20,
                          leading=42,
                          alignment=TA_CENTER))


def my_first_page(canvas, doc):
    """first page reportlab seeting"""

    canvas.saveState()
    canvas.setFont('Helvetica', 10)
    canvas.restoreState()


def my_later_pages(canvas, doc):
    """later pagse reportlab seeting"""

    canvas.saveState()
    canvas.setFont('Helvetica', 9)
    canvas.restoreState()


def story_effect(headline, rendered_graph, text, story):
    """add story for average and estimated_effects"""

    story.append(Paragraph(headline, styles['Heading2']))
    story.append(Spacer(1, 0.5 * cm))
    story.append(Paragraph(text, styles['JustifyAlign']))

    if rendered_graph is None:
        story.append(Spacer(1, 0.5 * cm))
        text = "Big graph not shown."
        story.append(Paragraph(text, styles['JustifyAlign']))
        return story

    story.append(Spacer(1, 0.5 * cm))
    direct_graph = rendered_graph
    direct_graph.hAlign = 'CENTER'

    story.append(utils.scale_height(direct_graph, 20))

    return story


def average_and_estimated_effects(analyze_dat):
    """average and estimated effects"""

    filename = ("Causing_Average_and_Estimated_Effects.pdf")

    print()
    print("Generating PDF report:", filename)

    # story
    story = []
    text_mediation = (
        'Effects on variable {}.'
    ).format(analyze_dat["model_dat"]["final_var"])

    story = story_effect('Average Direct Effects (ADE)',
                         analyze_dat["graph_dat"]["direct_graph"],
                         "", story)
    story.append(PageBreak())
    story = story_effect('Average Total Effects (ATE)',
                         analyze_dat["graph_dat"]["total_graph"],
                         "", story)
    story.append(PageBreak())
    story = story_effect('Average Mediation Effects (AME)',
                         analyze_dat["graph_dat"]["mediation_graph"],
                         text_mediation, story)
    story.append(PageBreak())
    story = story_effect('Estimated Direct Effects (EDE)',
                         analyze_dat["graph_dat"]["direct_hat_graph"],
                         "", story)
    story.append(PageBreak())
    story = story_effect('Estimated Total Effects (ETE)',
                         analyze_dat["graph_dat"]["total_hat_graph"],
                         "", story)
    story.append(PageBreak())
    story = story_effect('Estimated Mediation Effects (EME)',
                         analyze_dat["graph_dat"]["mediation_hat_graph"],
                         text_mediation, story)

    # save pdf file
    doc = SimpleDocTemplate(analyze_dat["model_dat"]["dir_path"] + filename)
    doc.build(story, onFirstPage=my_first_page, onLaterPages=my_later_pages)


def tvalues_and_biases(analyze_dat):
    """tvalues and biases in report"""

    filename = ("Causing_tvalues_and_Biases.pdf")

    print("Generating PDF report:", filename)

    # story
    story = []
    text_0 = 'With respect to zero. '
    text_1 = 'With respect to hypothesized average model effects. '
    text_mediation = (
        'Effects on variable {}.'
    ).format(analyze_dat["model_dat"]["final_var"])

    story = story_effect('Estimated Direct t-values (ED0)',
                         analyze_dat["graph_dat"]["direct_tval_graph_0"],
                         text_0, story)
    story.append(PageBreak())
    story = story_effect('Estimated Total t-values (ET0)',
                         analyze_dat["graph_dat"]["total_tval_graph_0"],
                         text_0, story)
    story.append(PageBreak())
    story = story_effect('Estimated Mediation t-values (EM0)',
                         analyze_dat["graph_dat"]["mediation_tval_graph_0"],
                         text_0 + text_mediation, story)

    story.append(PageBreak())
    story = story_effect('Estimated Direct t-values (ED1)',
                         analyze_dat["graph_dat"]["direct_tval_graph_1"],
                         text_1, story)
    story.append(PageBreak())
    story = story_effect('Estimated Total t-values (ET1)',
                         analyze_dat["graph_dat"]["total_tval_graph_1"],
                         text_1, story)
    story.append(PageBreak())
    story = story_effect('Estimated Mediation t-values (EM1)',
                         analyze_dat["graph_dat"]["mediation_tval_graph_1"],
                         text_1 + text_mediation, story)

    if analyze_dat["model_dat"]["estimate_bias"]:
        story.append(PageBreak())
        story.append(Paragraph('Biases', styles['Heading2']))
        story.append(Spacer(1, 0.5 * cm))
        table = table_bias(analyze_dat)
        story.append(table)

    # save pdf file
    doc = SimpleDocTemplate(analyze_dat["model_dat"]["dir_path"] + filename)
    doc.build(story, onFirstPage=my_first_page, onLaterPages=my_later_pages)


def mediation_effects(analyze_dat, individual_id):
    """mediation effects"""

    filename = ("Causing_Individual_Effects_" + str(individual_id) + ".pdf")
    print("Generating PDF report:", filename)

    # story
    text_individual = (
        'For individual {} with respect to population median. '
    ).format(individual_id)
    if "base_var" in analyze_dat["model_dat"]:
        text_individual += (
            'Based on {}. '
        ).format(analyze_dat["model_dat"]["base_var"])
    text_mediation = (
        'Effects on variable {}.'
    ).format(analyze_dat["model_dat"]["final_var"])
    story = []

    story = story_effect('Individual Direct Effects (IDE)',
                         analyze_dat["graph_dat"]["direct_indiv_graphs"][individual_id],
                         text_individual, story)
    story.append(PageBreak())
    story = story_effect('Individual Total Effects (ITE)',
                         analyze_dat["graph_dat"]["total_indiv_graphs"][individual_id],
                         text_individual, story)
    story.append(PageBreak())
    story = story_effect('Individual Mediation Effects (IME)',
                         analyze_dat["graph_dat"]["mediation_indiv_graphs"][individual_id],
                         text_individual + text_mediation, story)
    # IME table
    table, _ = table_indiv(analyze_dat, individual_id)
    story.append(Spacer(1, 0.5 * cm))
    story.append(table)

    # save pdf file
    doc = SimpleDocTemplate(analyze_dat["model_dat"]["dir_path"] + filename)
    doc.build(story, onFirstPage=my_first_page, onLaterPages=my_later_pages)


def create_table(data, align, fontcolor, backcolor, box, together=True):
    """create table for data as nested list of table rows"""

    data = copy(data)

    # nrow, ncol
    nrow = len(data)
    ncol = len(data[0])

    if align == "NA":
        align = zeros((nrow, ncol)).tolist()
        for i in range(nrow):
            for j in range(ncol):
                align[i][j] = 'LEFT'

    if fontcolor == "NA":  # fontcolor per cell
        fontcolor = array(['black'] * nrow * ncol, dtype=object).reshape(nrow, ncol)

    if backcolor == "NA":  # background color per cell
        backcolor = array(['white'] * nrow * ncol, dtype=object).reshape(nrow, ncol)

    # decimal formatting if int or float
    for i in range(nrow):
        for j in range(ncol):
            if isinstance(data[i][j], float):
                data[i][j] = "{:.2f}".format(float(data[i][j]))  # dec = 2

    # setting TableStyle per cell
    table_styles = []
    for i in range(nrow):
        for j in range(ncol):
            # pylint: disable=E1121 # too many positional arguments for constructor call
            # reportlab colors.toColor works as expected
            reportlab_color = colors.toColor(backcolor[i][j])
            table_styles.extend([
                ('BACKGROUND', (j, i), (j, i), reportlab_color),
                ('ALIGN', (j, i), (j, i), align[i][j]),
                ('TEXTCOLOR', (j, i), (j, i), getattr(colors, fontcolor[i][j])),
            ])
    # type instead of box != "NA": avoid FutureWarning
    if not isinstance(box, str):
        for i in range(box.shape[0]):
            table_styles += [('BOX', (box[i, 1], box[i, 0]), (box[i, 3], box[i, 2]), 1,
                              colors.black)]

    # creating table
    table = Table(data)
    table.setStyle(TableStyle(table_styles))
    # don't split tables over pages
    if together:
        table = KeepTogether(table)

    return table


def table_indiv(analyze_dat, individual_id):
    """table indiv for xvars and yvars, using median observation"""

    # header
    header = [[
        "Variable",
        "Rank",
        "Individual " + str(individual_id),
        "Median",
        "ITE on {}".format(analyze_dat["model_dat"]["final_var"]),
    ]]

    # dimensions
    top = 10  # top strengths, weaknesses
    ncol = len(header[0])

    # initialize header # ToDo: simplify header
    data = [[""] * ncol]  # initialize empty header

    # compute data, for xvars and yvars
    xy_indiv = ["x", "y"]
    for xy in xy_indiv:
        data_xy = []
        # define variables and data
        if xy == "x":
            dim = analyze_dat["model_dat"]["mdim"]
            variables = analyze_dat["model_dat"]["xvars"]
            e_j_indivs = analyze_dat["indiv_dat"]["exj_indivs"]
            dat = analyze_dat["indiv_dat"]["xdat_based"]
        if xy == "y":
            dim = analyze_dat["model_dat"]["ndim"]
            variables = analyze_dat["model_dat"]["yvars"]
            e_j_indivs = analyze_dat["indiv_dat"]["eyj_indivs"]
            dat = analyze_dat["indiv_dat"]["yhat_based"]
        # data
        for i in range(dim):
            dat_row = dat[i]  # row indiv data
            var = variables[i]  # var
            # rank
            rank = sorted(dat_row).index(dat_row[individual_id]) + 1
            # row
            row = [var,  # var
                   rank,  # rank
                   dat_row[individual_id],  # value
                   median(dat_row),  # median observation
                   e_j_indivs[i, individual_id],  # effect
                   var,  # var
                   ]
            data_xy.append(row)
        data.extend(data_xy)

    # sort data
    sort_dat = [row[4] for row in data[1:]]  # without header
    sort_ind = sorted(range(len(sort_dat)), key=lambda k: sort_dat[k], reverse=True)
    sort_ind = [0] + [el + 1 for el in sort_ind]  # keep header in first row
    data = [data[individual_id] for individual_id in sort_ind]

    # numeric data without header, with last col "var"
    indiv_table_dat = deepcopy(data[1:])

    # delete last col var and fill in header
    data = [row[:-1] for row in data]
    data[0] = header[0]

    # shrink table to top strengths and weaknesses
    if 2 * top + 1 < len(data):
        data = data[0:top + 1] + [["..."] * ncol] + data[-top:]
    nrow = len(data)  # with header

    # align list of cells to be aligned
    align = zeros((nrow, ncol)).tolist()
    for i in range(nrow):
        for j in range(ncol):
            if j == 0:
                align[i][j] = 'LEFT'
            else:
                align[i][j] = 'RIGHT'

    # fontcolor list of cell fontcolors
    fontcolor = empty((nrow, ncol)).tolist()
    for i in range(nrow):
        for j in range(ncol):
            if i == 0:
                fontcolor[i][j] = 'white'
            elif j < 4:
                fontcolor[i][j] = 'black'
            elif not isinstance(data[i][j], str):
                if data[i][j] > 0:
                    fontcolor[i][j] = 'green'
                elif data[i][j] < 0:
                    fontcolor[i][j] = 'red'
                else:
                    fontcolor[i][j] = 'black'
            else:
                fontcolor[i][j] = 'black'

    # backcolor list of cell background colors
    backcolor = empty((nrow, ncol)).tolist()
    for i in range(nrow):
        for j in range(ncol):
            if i == 0:
                backcolor[i][j] = 'rgb(40, 76, 88)'
            elif i % 2 == 1:
                backcolor[i][j] = 'rgb(222, 222, 222)'
            elif i % 2 == 0:
                backcolor[i][j] = 'white'

    # box matrix of boxes, format: from row, column to row, column
    box = array([[0, 0, nrow - 1, ncol - 1],
                 [0, 0, 0, ncol - 1],
                 ])

    # story table
    table = create_table(data, align, fontcolor, backcolor, box)

    return table, indiv_table_dat


def table_bias(analyze_dat):
    """table indiv for xvars and yvars, using median observation"""

    # header
    header = [[
        "Variable",
        "Bias value",
        "Bias t-value ",
    ]]

    # dimensions
    ncol = len(header[0])

    # initialize header
    data = [[""] * ncol]
    data[0] = header[0]

    # create data
    for i in range(analyze_dat["model_dat"]["ndim"]):
        row = [analyze_dat["model_dat"]["yvars"][i],  # variable
               analyze_dat["estimate_dat"]["biases"][i],  # bias value
               (analyze_dat["estimate_dat"]["biases"][i] /
                analyze_dat["estimate_dat"]["biases_std"][i])]  # bias t-value
        data.append(row)

    nrow = len(data)  # with header

    # align list of cells to be aligned
    align = zeros((nrow, ncol)).tolist()
    for i in range(nrow):
        for j in range(ncol):
            if j == 0:
                align[i][j] = 'LEFT'
            else:
                align[i][j] = 'RIGHT'

    # fontcolor list of cell fontcolors
    fontcolor = empty((nrow, ncol)).tolist()
    for i in range(nrow):
        for j in range(ncol):
            if i == 0:
                fontcolor[i][j] = 'white'
            elif j < 2:
                fontcolor[i][j] = 'black'
            elif abs(data[i][j]) > 2:
                fontcolor[i][j] = 'red'
            else:
                fontcolor[i][j] = 'green'

    # backcolor list of cell background colors
    backcolor = empty((nrow, ncol)).tolist()
    for i in range(nrow):
        for j in range(ncol):
            if i == 0:
                backcolor[i][j] = 'rgb(40, 76, 88)'
            elif i % 2 == 1:
                backcolor[i][j] = 'rgb(222, 222, 222)'
            elif i % 2 == 0:
                backcolor[i][j] = 'white'

    # box matrix of boxes, format: from row, column to row, column
    box = array([[0, 0, nrow - 1, ncol - 1],
                 [0, 0, 0, ncol - 1],
                 ])

    # story table
    table = create_table(data, align, fontcolor, backcolor, box)

    return table

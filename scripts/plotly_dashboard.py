import os
import re
import pandas as pd
from dash import Dash, dcc, html, Input, Output, State, dash_table
from collections.abc import Mapping
import re

# ---------------------------------
# Config
# ---------------------------------
KEY_COLS = ["chrom", "pos", "ref", "alt"]
OUTPUT_FOLDER = "/input/output"        # no trailing slash
QUERY_SCRIPT = "queryAndPredict_v5.py" # your script that writes TSVs
KVF = "knownVarInfAndLgModelPred.tsv"  # produced by QUERY_SCRIPT
SMF = "smallModelPred.tsv"             # produced by QUERY_SCRIPT
VEPF = "vepResults.tsv"                # produced by QUERY_SCRIPT
UNKNOWN_VAR = "UNKNOWN VARIANT"

# ---------------------------------
# Helpers
# ---------------------------------
def _format_clnsig(value: object) -> str:
    if value is None:
        return "None"
    s = str(value).strip()
    if s == "" or s.lower() == "nan":
        return "None"
    s = s.replace("_", " ")
    s = s.replace("|", ", ").replace("/", ", ")
    words = s.split()
    keep_lower = {"of", "and", "to", "for", "in", "on", "with", "by", "as"}
    pretty = " ".join(
        [w if w.lower() in keep_lower else (w[:1].upper() + w[1:].lower()) for w in words]
    )
    return pretty

def _color_for_clnsig(s: str) -> str:
    t = (s or "").lower()
    if "pathogenic" in t:
        return "#dc2626"  # red-600
    if "benign" in t:
        return "#16a34a"  # green-600
    return "#111827"      # gray-900

def _df_to_dash(df: pd.DataFrame):
    if df is None or isinstance(df, str) or df.empty:
        return [], []
    cols = [{"name": c, "id": c} for c in df.columns]
    data = df.to_dict("records")
    return cols, data

def _known_children(knownVar):
    """Render Known Variant as string (pre) or DataFrame (table)."""
    if isinstance(knownVar, str):
        return html.Pre(
            knownVar,
            style={
                "whiteSpace": "pre-wrap",
                "fontFamily": "monospace",
                "background": "#f8fafc",
                "padding": "8px",
                "borderRadius": "0.5rem",
                "border": "1px solid #e5e7eb",
            },
        )
    if isinstance(knownVar, pd.DataFrame):
        cols, data = _df_to_dash(knownVar)
        return dash_table.DataTable(
            columns=cols, data=data, page_size=10, style_as_list_view=True,
            style_table={"overflowX": "auto"},
            style_cell={"padding": "8px", "border": "none", "textAlign": "left"},
            style_header={"fontWeight": 700, "borderBottom": "1px solid #e5e7eb", "textAlign": "left"},
            style_data_conditional=[{"if": {"row_index": "odd"}, "backgroundColor": "#fafafa"}],
        )
    return html.Em("No known variant info.")

def _get_text(df_or_str, colname: str) -> str:
    """Take a DF or string; return cleaned text (one row)."""
    if isinstance(df_or_str, str):
        return _format_clnsig(df_or_str)
    if isinstance(df_or_str, pd.DataFrame) and not df_or_str.empty:
        if colname in df_or_str.columns:
            val = df_or_str[colname].iat[0]
        else:
            # case-insensitive fallback
            cmap = {c.lower(): c for c in df_or_str.columns}
            val = df_or_str[cmap[colname.lower()]].iat[0] if colname.lower() in cmap else None
        return _format_clnsig(val)
    return "None"

def _line_maker(label: str, data: Mapping) -> html.Div:
    """
    Render: Label: key1:v1, key2:v2, key3:v3
    - `data` can be dict or any Mapping (OrderedDict preserves order).
    - None/empty values are shown as 'None'.
    """
    if not data:
        items_str = "None"
    else:
        parts = []
        for k, v in data.items():
            v_str = "None" if (v is None or v == "") else str(v)
            parts.append(f"{k}: {v_str}")
        items_str = ",   ".join(parts)

    return html.Div(
        [html.Strong(f"{label}: "), html.Span(items_str)],
        style={"fontSize": "1.05rem", "marginBottom": "6px"}
    )

def _color_for_clnsig(s: str) -> str:
    s = s.lower()
    if "pathogenic" in s: return "#dc2626"
    if "benign" in s:     return "#16a34a"
    return "#111827"

def _line_from_dict_colored_words(label: str, data: Mapping) -> html.Div:
    """
    Render: Label: key1:v1, key2:v2, key3:v3
    - Colors only 'benign' (green) and 'pathogenic' (red), case-insensitive.
    """
    if not data:
        items = [html.Span("None")]
    else:
        items = []
        for i, (k, v) in enumerate(data.items()):
            v_str = "None" if v in (None, "") else str(v)
            # Break the value into fragments so we can color only keywords
            parts = re.split(r"(?i)(benign|pathogenic)", v_str)
            styled_value = []
            for p in parts:
                if not p:
                    continue
                low = p.lower()
                if low == "benign":
                    styled_value.append(html.Span(p, style={"color": "#16a34a"}))  # green
                elif low == "pathogenic":
                    styled_value.append(html.Span(p, style={"color": "#dc2626"}))  # red
                else:
                    styled_value.append(html.Span(p))
            # key: value
            items.append(html.Span([
                f"", *styled_value
            ]))
            if i < len(data) - 1:
                items.append(html.Span(", "))

    return html.Div(
        [html.Strong(f"{label}: "), *items],
        style={"fontSize": "1.05rem", "marginBottom": "6px"}
    )

def _lookup(chrom_val, pos_val, ref_val, alt_val,
            outputFolder, queryScript,
            knownVariantFile, smallModelFile, vepFile):
    """
    - Runs the query script (writes TSVs into outputFolder)
    - Returns: (knownVar, smOut, vepOut)
      knownVar: DataFrame OR string (UNKNOWN VARIANT)
      smOut   : DataFrame (1 row)
      vepOut  : DataFrame (1 row)
    """
    # Run the external query script
    pyCmd = f"python {queryScript} {chrom_val} {pos_val} {ref_val} {alt_val} {outputFolder} {knownVariantFile} {smallModelFile} {vepFile}"
    print("CMD:", pyCmd)
    os.system(pyCmd)

    # Known variant (may be absent)
    kvPath = os.path.join(outputFolder, knownVariantFile)
    if os.path.exists(kvPath):
        kvData = pd.read_csv(kvPath, sep="\t")
    else:
        kvData = UNKNOWN_VAR

    # Small model (required)
    smPath = os.path.join(outputFolder, smallModelFile)
    smData = pd.read_csv(smPath, sep="\t")

    # VEP (required)
    vepPath = os.path.join(outputFolder, vepFile)
    vepData = pd.read_csv(vepPath, sep="\t")

    return kvData, smData, vepData

# ---------------------------------
# Dash App
# ---------------------------------
app = Dash(__name__)
server = app.server
app.title = "Predicting the Clinical Significance of Human Genetic Variants"

CARD_STYLE = {
    "background": "#ffffff",
    "borderRadius": "16px",
    "boxShadow": "0 10px 25px rgba(0,0,0,0.06)",
    "padding": "18px",
}
INPUT_STYLE = {"width": "50%", "padding": "10px 12px", "borderRadius": "10px", "border": "1px solid #e5e7eb"}
LABEL_STYLE = {"fontWeight": 600, "fontSize": "0.9rem", "marginBottom": "6px"}
BORDER = "#e5e7eb"

banner = html.Div(
    [
        html.Div(
            [
                html.Div("Predicting the Clinical Significance of Human Genetic Variants",
                         style={"fontSize": "1.5rem", "fontWeight": 800}),
                html.Div("Two interactive ML models for variant effect prediction plus known effect mining",
                         style={"color": "#4b4c4f", "marginTop": "4px"}),
            ]
        )
    ],
    style={
        #"background": "linear-gradient(135deg, rgba(59,130,246,0.15), rgba(139,92,246,0.20))",
        "background": "linear-gradient(135deg, rgba(255, 255, 255,0.3), rgba(6, 90, 156,1))",
        "border": f"1px solid {BORDER}",
        "padding": "18px",
        "borderRadius": "18px",
        "marginTop": "16px",
    },
)

LOAD_STATUS_TEXT =  "Large Model Background:<br>" \
                    "  - Trained on a a large dataset including mutation type, cohort and constraint data.<br>" \
                    "  - Strengths: highly accurate (~98%)<br>" \
                    "  - Weaknesses: requires cohort data, so it can't make predictions about unreported variants.<br><br>" \
                    "Small Model Background:<br>" \
                    "  - Trained on gene-level constraint scores and basic mutation information<br>" \
                    "  - Strengths: can predict the effects of unreported mutations<br>" \
                    "  - Weaknesses: lower accuracy - prediction scores > 0.75 and lower than 0.25 are ~94% accurate<br>" \
                    "    Be careful interpreting effects with scores close to the mid-point (0.26 and 0.74)<br><br>" \
                    "Input: (1)<br>" \
                    " - info for one human variant<br><br>" \
                    "Outputs: (3)<br>" \
                    "  - For known variants: large ML model prediction and ClinVar's reported effect<br>" \
                    "  - For unknown variants: small ML model prediction<br><br>" \
                    "Suggestions:<br>" \
                    "  - Check probability scores for more nuanced prediction likelihoods.<br>" \
                    "  - Values closer to 1 or 0 are more likely to be accurate than those close to 5.<br>" \
                    "  - The large model is helpful when ClinVar's documented effects are conflicting or ambiguous.<br>" \
                    "  - The small model is helpful for predicting the effects of uncharacterized variants that can't be analyzed by the large model."

app.layout = html.Div(
    [   
        banner,

        # Status
        html.Div(
            [
                html.Div(
                    [
                        html.H3("App Usage:", style={"marginTop": 0}),
                        dcc.Markdown(LOAD_STATUS_TEXT, id="load-status", dangerously_allow_html=True),
                    ],
                    style=CARD_STYLE,
                )
            ],
            style={"display": "grid", "gridTemplateColumns": "1fr", "gap": "16px", "marginTop": "16px", "margin": "16px 0",},
        ),

        # Query row: inputs + banner image
        html.Div(
            [
                html.Div(
                    [
                        html.H3("Enter your query:", style={"marginTop": 0}),
                        html.Div(
                            [
                                html.Label("Chromosome", style=LABEL_STYLE | {"textAlign": "right"}),
                                dcc.Input(id="chrom", type="text", placeholder="1 or chr1", style=INPUT_STYLE),

                                html.Label("Position", style=LABEL_STYLE | {"textAlign": "right"}),
                                dcc.Input(id="pos", type="text", placeholder="1234567", style=INPUT_STYLE),

                                html.Label("Reference Nucleotide", style=LABEL_STYLE | {"textAlign": "right"}),
                                dcc.Input(id="ref", type="text", placeholder="A", style=INPUT_STYLE),

                                html.Label("Alternative Nucleotide", style=LABEL_STYLE | {"textAlign": "right"}),
                                dcc.Input(id="alt", type="text", placeholder="G", style=INPUT_STYLE),

                                html.Div(),  # spacer in label column
                                html.Button(
                                    "Search",
                                    id="search",
                                    n_clicks=0,
                                    style={
                                        "padding": "10px 16px",
                                        "borderRadius": "10px",
                                        "justifySelf": "start",
                                    },
                                ),
                            ],
                            style={
                                "display": "grid",
                                "gridTemplateColumns": "120px 1fr",
                                "columnGap": "12px",
                                "rowGap": "10px",
                                "alignItems": "center",
                            },
                        ),
                    ],
                    style=CARD_STYLE,
                ),
                html.Div(
                    [
                        html.Img(src="/assets/banner.jpg",
                                 style={"width": "100%", "borderRadius": "12px", "border": f"1px solid {BORDER}"}),
                    ],
                    style=CARD_STYLE,
                ),
            ],
            style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "16px", "marginBottom": "16px"},
        ),

        # Clinical significance (3-line summary)
        html.Div(
            [
                html.Div(
                    [
                        html.H3("Results Summary", style={"marginTop": 0}),
                        dcc.Loading(html.Div(id="clnsig"), type="circle"),
                    ],
                    style=CARD_STYLE,
                )
            ],
            style={"display": "grid", "gridTemplateColumns": "1fr", "gap": "16px", "marginBottom": "16px"},
        ),

        html.Div(
    [
        html.Div(
            [
                html.H3("Large Model", style={"marginTop": 0}),
                html.Div(id="known-box"),
            ],
            style=CARD_STYLE,
        ),
        html.Div(
            [
                html.H3("Small Model", style={"marginTop": 0}),
                dash_table.DataTable(
                    id="sm-table",
                    columns=[], data=[],
                    page_size=10, style_as_list_view=True,
                    style_table={"overflowX": "auto"},
                    style_cell={"padding": "8px", "border": "none", "textAlign": "left"},
                    style_header={"fontWeight": 700, "borderBottom": "1px solid #e5e7eb", "textAlign": "left"},
                    style_data_conditional=[{"if": {"row_index": "odd"}, "backgroundColor": "#fafafa"}],
                ),
            ],
            style=CARD_STYLE,
        ),
        html.Div(
            [
                html.H3("VEP Result", style={"marginTop": 0}),
                dash_table.DataTable(
                    id="vep-table",
                    columns=[], data=[],
                    page_size=10, style_as_list_view=True,
                    style_table={"overflowX": "auto"},
                    style_cell={"padding": "8px", "border": "none", "textAlign": "left"},
                    style_header={"fontWeight": 700, "borderBottom": "1px solid #e5e7eb", "textAlign": "left"},
                    style_data_conditional=[{"if": {"row_index": "odd"}, "backgroundColor": "#fafafa"}],
                ),
            ],
            style=CARD_STYLE,
        ),
    ],
    # Stack vertically with space between cards
    # App background color change below
    style={"display": "flex", "flexDirection": "column", "gap": "16px"},
),
    ],
    style={"maxWidth": "1100px", "margin": "30px auto", "padding": "0 16px", "fontFamily": "Inter, \
        system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial", "backgroundColor": "#f3f4f6", \
        "paddingTop": "4px", "paddingBottom": "16px",},
)

# ---------------------------------
# Callbacks
# ---------------------------------

# Clientside: disable button & show "Running..." immediately on click
app.clientside_callback(
    """
    function(n, chrom, pos, ref, alt) {
        if (!n) { return [false, "Search"]; }
        return [true, "Running variant search..."];
    }
    """,
    Output("search", "disabled"),
    Output("search", "children"),
    Input("search", "n_clicks"),
    State("chrom", "value"),
    State("pos", "value"),
    State("ref", "value"),
    State("alt", "value"),
    prevent_initial_call=True,
)

# Server: perform lookup, render all results, then re-enable the button
@app.callback(
    Output("known-box", "children"),
    Output("sm-table", "columns"),
    Output("sm-table", "data"),
    Output("vep-table", "columns"),
    Output("vep-table", "data"),
    Output("clnsig", "children"),
    Output("search", "disabled", allow_duplicate=True),
    Output("search", "children", allow_duplicate=True),
    Input("search", "n_clicks"),
    State("chrom", "value"),
    State("pos", "value"),
    State("ref", "value"),
    State("alt", "value"),
    prevent_initial_call=True,
)
def do_search(n, chrom_val, pos_val, ref_val, alt_val):
    # Validate inputs
    if any(v in (None, "") for v in [chrom_val, pos_val, ref_val, alt_val]):
        msg = html.Div(html.Strong("Please fill all four fields"),
                       style={"color": "#111827", "fontSize": "1.05rem"})
        return msg, [], [], [], [], msg, False, "Search"

    # Run query script and read outputs
    knownVar, smOut, vepOut = _lookup(
        str(chrom_val), str(pos_val), str(ref_val), str(alt_val),
        OUTPUT_FOLDER, QUERY_SCRIPT, KVF, SMF, VEPF
    )

    # Known Variant box (string or DF)
    known_child = _known_children(knownVar)

    # Tables
    sm_cols, sm_data = _df_to_dash(smOut if isinstance(smOut, pd.DataFrame) else None)
    vep_cols, vep_data = _df_to_dash(vepOut if isinstance(vepOut, pd.DataFrame) else None)

    # Clinical significance: 3-line summary
    kv_clnsig  = _get_text(knownVar, "clnsig")
    kv_pred  = _get_text(knownVar, "prediction")
    kv_prob  = _get_text(knownVar, "probability")
    sm_pred  = _get_text(smOut,    "prediction")
    sm_prob  = _get_text(smOut,    "probability")
    vp_clnsig = _get_text(vepOut,   "CLIN_SIG")

    clinvar = {"Reported Effect":kv_clnsig}
    kvDict = {"Prediction":kv_pred}
    smDict = {"Prediction":sm_pred}
    #vpDict = {"Prediction":vp_clnsig}

    cln_child = html.Div([
        _line_from_dict_colored_words("Clinvar",clinvar),
        _line_from_dict_colored_words("Large Model Prediction",kvDict),
        _line_from_dict_colored_words("Small Model Prediction",smDict),
  #      _line_from_dict_colored_words("VEP",vpDict),
    ])

    # Re-enable search & restore label
    return known_child, sm_cols, sm_data, vep_cols, vep_data, cln_child, False, "Search"

# ---------------------------------
# Main
# ---------------------------------
if __name__ == "__main__":
    # Prevent hot-reload resets while long tasks are running
    app.run(host="0.0.0.0", port=8050, debug=False, dev_tools_hot_reload=False)

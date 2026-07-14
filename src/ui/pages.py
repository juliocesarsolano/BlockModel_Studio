from __future__ import annotations

import base64
import html
import io
import re
import uuid
from datetime import datetime
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from plotly.subplots import make_subplots

from src.core.cleaning import clean_model_data
from src.core.comparison import (
    category_comparison_by_role,
    common_roles,
    compare_global_volumes,
    compare_model_summaries,
    incompatible_grade_units,
    pairwise_volume_matrix,
    volume_gate_status,
)
from src.core.io import dataframe_to_excel_bytes, load_dataframe
from src.core.metrics import (
    PERIODS,
    apply_categorical_filters,
    contained_metal,
    grouped_summary,
    model_summary,
    negative_counts,
    period_summary,
    tonnage_column_name,
    tonnage_divisor,
    total_tonnage,
    total_volume,
    weighted_mean,
)
from src.core.models import (
    CATEGORY_ROLES,
    GRADE_UNITS,
    MODEL_TYPES,
    TONNAGE_UNITS,
    VOLUME_UNITS,
    CategorySpec,
    GradeSpec,
    ModelBundle,
    ModelConfig,
    Scene,
)
from src.core.parameters import cleaning_defaults, column_aliases, display_defaults, grade_defaults, master_filter_defaults, validation_defaults
from src.core.reports import scenes_to_excel, scenes_to_pdf
from src.core.validation import validate_model
from src.ui.common import CATEGORY_COLORS, MASTER_BLK_MODEL_OPTIONS, MODEL_COLORS, format_table, move_to, selected_master_blk_model_values


ROOT = Path(__file__).resolve().parents[2]


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]


BARRICK_LOGO_B64 = """iVBORw0KGgoAAAANSUhEUgAAAbkAAABaCAYAAAAl8yiwAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAAJcEhZcwAAIdUAACHVAQSctJ0AACmoSURBVHhe7V0JmBxVta64geITRVyf8nxPXFB8z6cfbqhxAQkQSWa6qnomW89MIKwReOwkk1q6eyYJmemZLEAIJDOZBEnYeQ9UFhVQQUU2BUFWQdmEhCSYIAmZd/5Tt0JnUpO+1V1dVTPe//v+r3um6966de+55z/31q1bmoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCQhWYYL1bm9T5Hm3S2bsh/T7eeodIoTDiMdZ6S+VGj4J0DmPu3uKsox+Tzg+oA5+ob2LaMeW8veprG5T3ZOtd4mzpQd37hMgb50kbjpr3L3LXTsdMOW0vkSp9QPmarA9rhvUDzSxYmm6DN2iG86Kmuy9pujM8TfrdsP9Ex3fRd0rnnKllrU9ynnEB/UK2HeIWZOOsveXKJngkUbf20abO3S/WOtyBxvw3teZOatwKDV8rYTi6/RzxCmFwlmd8zpEsCDNmvFWUaGTDMN6s6XOu9eo0oB58Ns99ScvMaRep0gnduVibPC+4/FHQs4m/0PfL37AJkBxLo3MYdwh0kLhhWF+PpU/ozq/JmS5549rhUO2ZdH5yBtQnkhBB3b6Z6j2orDtz8nwSAudCkSp5ICAziwdyHRrkY8w82ZfzsjZp3qA2tWtQm7JgUJs0d1DLFuTY1EFpzvPSTZ4/qBnu3/m6s/lLqI6Op9/eL85cHxj2nRV9CDiF2kF35otU9cWhFNQYhXlUt897fTegPEE03PXUNlu0bHGt1mB9QeQWI3TnUG0yNWa2uGtDR046B4wOhgPC+EwXv1FluL8hw7K0jPvf2riZe4jSjTzo1iS+tqYK9dlMHa65YyNd96dEyvQBAcnU7uDyR0XUU7lN+DTILnyRMd2LyKGerU3p2p9KNcYrXB2hW9+JpU80d3oOtPy6UReGvZHqha7dvoP7hO7qHD3HAd25h8sRVN5yTi1R+ziXiVTJQe/Yn3xGnuzjMc3Ib9lRh345YUeGUzv9/GAXEEszv57a5kLqw18VJYkWuB62QXHe4TiN2iETQ7Dx/Vn/TgJ/B9dvJd9WTg4yyIeY+e7k/HrG+S53tqiMISzNvFcZKAOLXv5V+vteMthjKJLdU5RyZAAjD8N9lAUs6Fp3ok1OjYxYt64QqdMHODGOYoPKX2f6dgGiDJ7ovEJ//5xtA1FlvZCxvpVcn+Cgb+c+Aadius/SbyVNL3xJlLI+0O3fsBMPLFsZ0R6Gs0qkihtjqH4mkD38iOzkFc/x0sgrSzYTVNbIKdoI/bepY6uW7fwR9RVDlC0aGM4fPRsMOn8Z2Yc4i0Wq+qDBOYhs8DE+V1AZhiP6rVl4RdPzx4ucEkLSIjeUcG4oDzqRWfwtVa4pSpp+ZOY4oQzBu9btWqPzbZFDupCkyAUR9YXIkDuP+4iWdU6vS3SYqMgFELMdEDrYVrbwGonQAPXbb4jSRou0i5xZmEZ18TPPR2DEFpewDUPYJOwR5WnquFib2P5FUdLakBaR0/PTqc43eDZBgXlQGYaS6wT2YV+jTTj38yKnBJE2kSsnRkQsdvkBzbDeJkqcTujW/tT51rEzCrqW4TiJOkjGuiuV15c2kStnM0Xu07rpe+FWbcLsz4kSR4O0iVw54UAwcjHzWzTTOTPy2Y60ilym/RvU5pdzkANC+IPKlRRRHoy6s8X1JAoZUerqkbTINVr/QddxI9d1GJ+GEbUXhBZFTilAmkUO9I1Hd26hBk3vaq6MdZk3ipOMdsoJo8jYx4ic0oM0i5xPlC9bfJ7vo0WFNIucT54uI7EznJ9Eer8ujSKH0US24N1vq6VNvPv/guS40cY8M0DXws4cU57idwQTQXlUIvJgUchfpzVYB4orCI8kRU63vkTX8TjnLR1M0HE7+qJ7tMgpJQgrcpgigLGFIS4eDc8GVKXxTKV8cIM5joUHYaE7YynK3FZ1x0D9Zwt/pk5R3xVbYSEtciTsaNegtpchHIy/yKOaOoRTbup4QdPbx4qS14awIofyB11XJcK5ljvWoLx3Syof8jGc5Zqx9s2i9LUhTSI33vo0jdR/6rVviNGET1/UkN4bZUEQ1tFv6+g6yV7su+nvNeQDF9J3rPq+n/7GqlE6xt7OaWAHYW0S52tZhMD1WnEl4ZGUyJlOhur6Zc8GQkxPeuVYzQtUUocwIoeLwbJyw7FDUXcuIgN6gowOBvYqd0yZBiwnzo2KN/OHiJKnA5hm1K272dEFlVuWqJOMPU/kmg7IipznoJ8mBrf/7qjbDvEq4gv09zp2TKgLOKSgcwWSOiN3Sudlao8PitJXjzAix33CvoS+B1/f7qhbPdTmD1Ae5HSdDVzXsCPp6BmkYzFtq9unidLXhrSI3MT2r2pm4a98nlD1QcTxuAa0YbYAQfsxfbeprpupzvdh4qHwobcIctaevHiMj0HgOg/2+Uty+tu8UUqFQARCDGffVHyE2rS1psdf4ha5hnM/RPks5WtA4BV0riDyLSUEqQUMQFIKWZGD4aACJnZ8UqQMB8N6p6Z37aM1uP9FQ3l08t96IzxUqGTEAAeg2zeJHNMB3T5WbrRTgXDqTZ0btImzDxA5Jw9ZkeN2cW4VqaoDnA6cS8Y5nOqhi2ziGU/sQoxwuMPbvZplvUnkWh1kRY77BNmvbn1GpAwP/dS3c7/wHlw+m855DecpIzQ+0S8xAhlv7SdyrR5pEDnUp5FfHz5wpPbwRAa8lf6eoU2nuq0FGCGbxYOpPP2U3xbuD0NHdpjFwKjPcJ8jtmvjItjgIE6Ry7R/mfzPn7yAYsi1DUfYvif8D5DNZEVOKUVYkcs4XxYpa8OMpW+lfI+jSn1COnJgh+L8RdOL7xO5JAtML5r5Z3kRRFB5fcJwKjprEnoYbGbO1SL35CEtcuQUdftXIlU00Av/Snl3sGORFTocpztbtMb8v4lcqkNYkctYXxMpo0HWHU/X8otQTh7T+bpTEDlUj6RFrsH6CNX770ILHNoLttrcuVqbXPyuptUY6AQh436W+vJVbGdod3b0LDLbaNTZrU20qhsABCEukcu0TyC/vjFUffOtCbS/u5yD09QjKZHz0Wh9kyptm/S9Ona6FKGlARlroejswxP1pjubiZv4e9AxPiGGGP43pmRKNkmR86FbJ/B0SKW6Y8JGyZZrfWYpaZEDsFWT6d4j7Xwm4bpJoJqtfUUO1SFJkeOpf+fX4j6jPFGWbPFeEp8WkVN9YTrf17IdG7htsoU12kQnmnvB5ai3yOHRG2ywwDNIFYL0cuKauVx5W+Q0ApC0yAEZeyk7sqDzDiUiVtOeKVImB7PwFTKO1yqKM4/OnNPJEY6Xclg4hm+Id71dnCk5pEHkAGw1JdPhQa/+fiJSVoc0iBzQ2P5tyn+DVABoEiHwhlXbc0lJilzjnJKUvflE/bMgumsjmSIMA7R5U+eR4q/oUU+Rm3jul0jcfs2PL8nOkqCu0TbZ4q9oYDJR5DRCkAaRa5z9CWqozVKdmRvVOkmkTAiDY8gZ3FRRtDhCcp/kqJzn9t37pHZD8a7xBHGy5JAWkcOUikyAAHoid6NIWR3SInKAbj0o5exQFl4E49b2zGBSItfQfpAXNIZ0uka+JHIYXaiXyEGgssWNXt1VsG+faBMOJvLLdlmsMyKQBpHLld5NnWujlIHjBq/pnCJSJoNGq8VzuhXqzDOkN6ZWZRep4B6fWXyKDKr2lYK1IDUiJzkKBj2R+7FIWR3SJHKyzm5ki9wYqsOfS9kak66Vnbs7S6QffaiHyJn5mWyvHHwH5BVE9KdsYSulnS1yGYFIg8gZ8z9InQv7Egaf2yd+x8aotU7J1ILx1r6aYT9Z0QC9On2MX13iA0uK8QqPik4Ei1AocspYPSJlMkjNdCXeVPFPK3KjfySnWzqfE9cQdK6h9OztlpE5qpBElCKHFdvoy6i3MCNl+CDTuUXT2+uzCXVsSMU9OetcaoDtFY3cc2B3iFTJAMIDwwoqXznZ+OxzRKo3gGd1ZDo0jLGpYxOv6EoK6RnJNYYTOae2x0xSNV1pPzSqRQ67teiO/GpK+CDTfV7LjYRVfTUgKpFrcBrIRvFqpMo+xyfqGPma+aU1P46TCiQtchNmfZTyfrzyMnw6PzYzxtRVUsAIMlvYVDEawn03XJN+zq6POsBoMvZdUvfm4Eh0+3qRMn6kQ+TGkOP+hVSHB71A6DqRtjqkZuGJNZHyf5XFK+j85URZsFIuY/2nSF0d4hY5s/0gb3o+4BxBhD3inXujHVGInD89Kdt3QPQfbMhsutFsLpAKJClyeG1IU/Fhr1NVeCAc703Snat50UdS0K0fSzl97ojWNJFqVxjWDKnICs4NbWM6h4mU8SINItdon8yrwCrVFQhHifqCTdeCNIgcL1Zyfy8lOKDXBrfVvI9l3CKHHZRkr9Fz1vel8o3qUaMWkeM3cbu9oVdPeo9uXE++q/rNDVKJsCI30a1tRw688RijsaaOK6kBXvPOvTuBQ+VjNwHnZ9pR1odFLvED9w2wRLuSs0UklLF+s9uOOK53D3ImcvdavJHJ79npxY0kRW7i2e+l88+h+t4s3VFhn4bzTM0LdpIWuabO71A0fbv0FB4IZ9dozRE5VI/YRc66Tfo62RateJ6FSxrVilyjdYRmFh+UCqJ9YiQNgcu66XnTe6QII3JY4q9bBU2f00qcvlsaVhsdezRF4mdopni9v26vIUf0Eo9QuBEqTMOgXLyaMr+a8kvuJjNe0Knbj1c0OtQRP8ztThIph4dhmdS5X69Y7yAbshX/ilJpkYMQ17itF+qYXzprfZ2Eg2zFedjreJICB04jW9HtLpFj9QgrcrVGvrjupuIHqOwnk7hdytcsO7oBIe668zequ9pX48Ypckb7F7Vs8W9SbQxfkS1u0EzrCyL16IasyPHG9WLPW6PwA7ZHmVshPtG/s4XHaKCRvregRAZZkfOJDgXlR5rdUhyDSoSzAuEMYdC7Ezd/io4dXP4e+vtYUdLkkHHOlXL2KHfGJoOREGTcmzNcuSgWhpstPqPp5/6rSB0PZEWOnaz9FNEPZsIxY5foXNjU9iX63MaBDa456FyBtEXQ5DxEglPbXoVAGJGDPWPKLei6ZGjYN2i6i53vvb0a+Tok+yKIY3kqP6L7VHGKnG41cT8Pyn8oUS+6k9z96bghK3K8lsG+jWx/MbebbFAIP+vtt3mddtQZyc2QxYKwIofj0LFkGZhHAOEoWQTzm6kB7tOM/Ina2BPeKUqZHPDyQMN5hW/qB5V7B+laUf6MnRMpKyPjYjQXkNdQikcK9DkXiJTxQFbkUD6M8v1gJizhMFnIKwRAwxF12Nz5HNlyNPeLZUXOJ84fdF0y9B0TO6cQ/QVEXXH7uKUR+aod3WmQs38ii1yNC4pGEmRFTqe+h77j2cGuvweSjssWyae5C8XZRjlCi1wdyNGwfQeVo1ebVEjB69J3YAx1xD4pR8+OgUZmhiHvbLC5acZ5QMqpoI6ayDDxFoe4IC1yCZIFsviopkf4dvCwIpcEERSibcx8tyh1NEiryOE43b5GpBz9kBW5asjTmfnaniUdUUiDyGHEh3sKZv4hMuQrqDzn0t9H7vQgdRJosL6jNXW+XnF0gfIjmqrmpZ14SSE6sMyoF04tY98iUtYfaRY51Dem6cz8bdqEsz4mShwN0ixysBOMALOFV6mvRD+VH+s9OetsaZFjx2wvESlHP6oayQX8HkT2V51b1UgubkJIMFpBB8NcsTd987SmU6Ra67M/1QAjMtl7Zmxg9k9DjeJ2YHAMC5fMBtUwTrQVIuA4kDaRg43A2WEhDp5DNPNnamNze4rSRoe0iRzaHdPlEBb0i+bOy7WJEb0FfShiHcnZXVLnApXIBZODa+d2Yh/7Kpn9f0HYlHdP7hrexWlUI6zIoZNVS1RsUJ5B5GiDOjYeHzDdl4kOLyuPC7rdys+ZVKwX+h0dtRbhwai1uaPyji8gv1LFeVSbctpeInX9kLTIocOiE2Pk4jn4zZpZuJPEbWYkC0yGQ9Iix6JG/QVOixfh8P3gFzWzuJTErrZnACshVpGbc4a0yKEvZuxrRcrRD1mR4/cIup2cRndcTuPZixwn86zAfdSfvsN5jEqEHsnZG6ky8br+kLRf2fE8Bt9wR9QhKXw4Dq/4byreG8ubs/FAreE86a1cCihPObnzWb+SWlG5O2QsuV1QQDiYRnIQ9UYokROOeXeUDnLoON3eRmL2HNnbH7kTG1jhOu9TsTwIHFbkgq51KIPSBZLPuYX4ItXB5VQOmz6z/JhBHIhT5LCji+x0JexQd9XCk6HErEambISLTSiyxS1i5Lvr8UHEsdn8ds0o5EUuowyyIuePrHTnUI6iwxKrFPn5Ocdm6s6l9LmOHQAMXcYBwtCbO/7Cz1LVE6ZdZCEOKkM5efqscys5xdofBsb+jM1zydAk6gHtgOcNs53R3osainAjua1eMLMbyjp81GtTcTM53KmJ7J0XeiRXMfDbLOWwQIxcTfcubXLM70fzEafIhdmTlEXO/j+RcvQjjMgN3fGkcdY3yU/+1eu7kkKHfsk+L3+1dtQZya6FiBxhRa4hwocxWQCdBnJqN/I9B3TwoHPvIDUYvyXaeYHT1gN4hT0iaRlnDCPK2CtEytqhOz+Sm76heuAIbk6/SFkfyIocygznWB7UBNJupXbeziIWlE85YWvZwotaZvZ/i9LEB1mR4z5BNqtb43a91jJm3QPo2N/yVHNQPjuR8mzu3EZtu0yUJl7EO12ZFflUJqaus4V1FOCmafV1/VCLyAGY8coW7+ffg9IFkmwPQodbAo2z/k3kNAoQVuTq8RYCoLlwGDmMZ1jsgs5fTo7qnJWUKup9LLEZsJxj59FGx2atsf0wfhRg0tnvqYmI3BHZ4vplRrU8iuygUaTzDVH26BFO5H4hUu0emFqRdWw4N0Y1cW/nFlbk8MLPStCt/en4zdyHgvIqJ9oWI5wwz1xGhThFbmL+41THT0nVCTgF/d5tEqlHN2oVOeD753yARmY/FQtMdk07HNG2TcVHRs/uMmkROaCxPaM1z9tauSzkBEyMtCKO6kzrEKlHBphUxoy9jZzCi2Rk2KmjdhqUV+C5hiFGtRn79rpN6YUTObm9K/HQsuH8wctXYiplajc+/1ekjgdhRU52urqBRrIs3BL2xTMbRARRcSJOkQN0+1YpGwPRJrr9O5FydCMKkQOwVsDMD/ACFSm/BlK/RJCFV/Qk+daXyJAmkeOtruw7pebp0clwXy8qeM73Dul7BD79+0xRMegcwxFtwuWt06a19RA5YMK5nydbepEFIii/nUjXyK9Ysr0VZHGgXiIHZKzlUvd7QZQhW/yT1mh9WqSuP2IXOet/pBdcefX9+uhwvBUQlcj5MPNzuS9L+xgSOrY/HO/mtXEz9xA5jUCkSeSAjNUh1bg8nRPBqkYf/utvKtVD2oi60p2ntCPrsPquXiIHZOacymWHXQXlWU50NDhCTOfGgXqK3NSO91JauTdQwNHAiZnunSJ1/RG/yO1PDvg16ee7vLL9UpuRwFs54kTUIgdknZPYXqWCS0H0PcwYZYu3xrrbUqRIm8g1zf4sV2wl54ffs4Vt1MC1PxTrvdblL6EaP02Ew2lsnyuuJjrUU+SADDlJTKME5TmU3DmLG7WJdV5ZC9RT5ADsvo++JBtVewEd7kHXH3GLHB4JMZ0+dtZB5wkiRsJ6jCP7JFAPkQMQKDZ1bPLaWOJ2gU+0d1PnC1qm/XCR0whC2kQOjwfIiBwIB6y7R4mU1UN35kpPIaWRqC8IQMb9rLiiaFBvkeNX67gPS08Re7vC3Kvphfq+jaHeIgfo1gneSmEJO/cXouj2dJG6fohb5IAG5yCuB+ndOui4puI2rdGaIXIYfaiXyAGZWV8ju/2zVN8uJ8rDq6Pdo0VOIwRpEzndXh3K6emFI0XK6mBaB1Jer0pH1akkRWQw2Ix1hbiqaFBvkQPM9oMpQvy7XP3TdWKlmO7W93mpOEQO27np9vXSjoYDGT7XeJFBfZCEyAG6JbcRuk/URRYjexoVJ4mxY+uzOUE9RQ6YePYnNbN4T+jgHhoA/2y6lshpBCBNIofXthvuE2zAQWUop3fMwzTyq+1Fkbp7lVznIgcLRwOjiJOoc2zCGlimMiK6xQ4t2CIsKsQhcgDeaM2LDyrYIJOOwUIU3T5PpI4esYgcQT/nfXSOp6WcGYjj+I0LNb6kdXdITuT2ofqs/GLiHaQ+4dXHs7wqOgkY+Q7qH3dqZscRka9wrrfIAahzw7me85BeeUnEsZzGuWRkLEhJk8gZ+WXSozhuXPtGkbI6YFuhJsk9I1ls8PZlsWNLXNRtyUUKRBYb625tXG80hheXyAG6cxOLOpxX0DnKiWAD5WqYXZ9npuISOQCPCOBcUlN1VDewe8Ot3zL6pEQOaLSO4P4vPasihM5wX6e/F2rjKGiIA0ed8xmy19VcB5hN4lFl4XYt2/ktcUTtiEPkAF7Rnl/CQSZ8XNA5ggjbZ6Fzb490g5C6IA0i12ztS+e40nNyAecOIndE62yRQ3hgVSbu78h0aBAN2mifI1LHB8NuFtMDweXaiXQMnH/GPlWkrg1xilzW+hhFiI9Jt4e3Q84mLdMe/YbFcYocoNNINpTtkz1k7NUidbRIUuSAjH0W12mYkQUCBNSfWfiDZhbr+7B4Nj+dbG+D1y+EfcAOUGdNxe3Ey/gVXbUiLpHzodsnk1D/Q+qc5eT7yoX19P0HIqcUIqzIRaXa4613aNm5B1DlFuncT3g7/gecN4iI9HT7NW3CrI+K3MIjY50u5cBBRDm6+wjvbBI38Poefc4vpUe47CDcZyNZnBGnyAEZZwLv4iK7AAFlM/N/0PRzo12IErvInfp2qr/bpQUeAoBrz1jHiRyiQ9IiB+hWN2/ILhXYlRFthn7S1Hk99Vc90vdRTpp/MLX1TXxLAG0edH5uF64X+Mql5CsPFKnDI26RAzLW9+ganthJwGWI+uC+6CxNZK/Ziggrco35QwK3papE3EdApzQLFkVreepMD1GFbmajGM5ogohy8OID26q6Qo25+1Fe6/h6gs5RTpwPHQc7wSeFjHO4/E4sRNRpxrpYpK4ecYsckLHmy49qMH2HY92bRepoELfIAQ2FD1H7bmInGnSuofQXojTMrn11cTnSIHIIgHVnrTc6k7T5HaQ2gc16PuVpqs9u78XEne8JNY1/6Gl7aQ1zP0J+6xRq55+Qr9gmPaOCMrNd5rdQ313Oaw3CIgmRAybO+jjld4snWiHqHvXC7eX+iOqsfq/BqgqyIudTdzYQg7el2i1tbyd2PBeFhoERykbsPv2KNKgD5KzqX5aZsft3RFyViBEmHHjSEQq2P5IdzcEB4nUbtU4tJyFyfI/A/YV3Xon7c7BbPj9F/1EhCZEDsK1dE51Xtl+gjNni41pmdnSPjqRB5HwYbontQHZHlHJy21AQC3/B/aHwEvX7++n6LCZWB+rOSVojCSDepaY7Bv3/NO//tkN/P0JOfgP3OVlx24l0PN5cbzi/1yZ2hH8PZlIiB2D3J+SJupe+PyqI+m4q3kV1ur/ILQUIK3KewVRH2XMEkaMjvEA1f21NCyvw8DjKIhOl4Jjmjm3kxL4nUicHvOKIN2+WdIAsznNuEamrQxIiB2RnH0Ad5VmpTg6iTuB0G6xpIofakJTIARnrfHZcQefbhWIhiu7eI1LXjjSJHGC4JvXX+/l8srY/lGgn9Pny91mCuE72S3QMPmHr/m8chIvfwtLPC68Gqva2QZIi50N3T+Z6AIPOPRxZHDuejLRf1ISwIpcEEUmhok2nUNMbscfS6E935fbGBHFcZs6tInXSGEOR6NXS9y7RsXkrLLtZpA+PpEQOwDZr2Q75KVp2Yp2beIVerUhS5PAuL925J9ToBfWfsS6j1LW/lSNtIgccctbe1PfXUF1v53oJPaoajpQP8oKNRZUn+yoEo+5povTVIQ0iB0ycjTej0IgWNiEzswLScV4AsVkz8ieKnBJEWkUOhgeDhpNt6riXyjlBlLh6YJcJWYHzHNh2ch7jRerkYRQ+r5kF+YUZaFfdebTqF3AmKXKAbl0k3kIgRy6r+0ca0X1E5FAdkhQ5gDcocF9nZxl03qFEOXDtjdZJIofqkUaR84El+mbhbm4b9OOohCkKYpSINmjuvE/T22t/VjUtIgc0WF8hWxQLUgLKMBwx88S25F5X0yLBmpEWkYPBIhqH08B0Qbb4Ov19F0cCUTxwiGkDw31e2nHwiMm+OXWrhWDQ0tNZRDZMyxapwyFpkcNqVjwTxmWQiSIxfUe2o7ty77YbDkmLHIC3oqNeZR05Ah+UpWGOIXKoDmkWOQCP/jQVGkhUbuG+jL4Av5GE4PntD5szC08QT4ns4eg0iRzQaP0H+2Nca1A5hiPqCP23qeNxLdNe392yhkUcIocLxcgMxlhONhBqJKyWxHcTb/61H6QOa2vN+a9rhhHNGwYAjApkxQEOI1vEvbiUzCmXATu8GO7fpMWaj3M3aI3uJ0QO8kha5AC8kLS542Xp6wV5mtlaKHIIjzSIHKBbV0jPPIAoc7b455qWrqdd5Mphdh6iZTsWUznW87X7ghd2QVsYwo+xuCIQp3MZ+SfITs6K/EH0tIkcgFWiZv4GT9RD6gVsCvvrYgOO2IHl6S0LqbJIaFD4yEmN8EZjbSSup0YhMXPXUYd6hNhFv2Nnj6na9K59+JmhqIGtriCkePYmsIxD2LYYhrNCpE4f8A4utBmvVK1Eqv+2JRChm0KvSDXsG7TplDYw3zK2LkL+D4pU0aORrhfTlmjDoPMPJY6dfgHKdA4/ZxgWWOQj0ydQnlwvnae99jdhBAFL6fGAfAudI+j8u1C0NZ6TxAYL1QB9Eu0ZmH8ZUb+GfZ1IlSzw+qKMfYxmFpdS+cm3OFu4LlBOTCOWB9ZDne9whBP30/gjRuRnuFvpHM/Q/87TTOcwCjrfKUoRLQznec8Gy+o8iEfDzp0+kSoOjCFhX8a2L9sfmVR/6CugQYMY2HZswDAUIoMTB20rVRMpT9OZTUZxLDsOw9rPe62NRWJGglbLIpIwMJ3va5MX5IPLOIQYRRpuO0XD7xep0wdMiWBqpLm4a/mDmKXjMnb4a8Ky6snnBedZzuZ5FKRY9d2Z3HDbPDsNOH8Qm4oulenMqoIm7L4i1Sfo9ywdl83X736DOfu/uH4Dzz8MJ8/Pa2aVmzZgwY/M+WAXeP4sbWDfYn2GBLidAoQC+Z576JMCagqsdRI/zEwEOuEh9GYO1pPYIRh/kuyvxP3IdA7SjLl7i7PVD3i8Qcbe0Q5R7lcrC7N4bKj+6BNt0tTRWdc+o6CgoPBPAwSECKghfnieEDsdyQQv+pzJnmBSWqx4VVBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUIgalmW8bS0Rn729M/fo6tLfvmJFbs819LnyvCl7gYsXn/DOgd7J7xqwPK4o5d69+vxJ71ndOfLY3zH1vT6XLmje95Le1vf5pGt9/7Kepg/4XDE/98EVi4ml3MdWLcwd3F9qaVze03Iorl9Un4KCwkjDzyzrLXB6cG7nk1NAh94dryYGOZO4uKxr+j6XLmr78MXz2j480Hv0R1Z2Td8f7Oue9tlVXS2fu7Rn+n+uLLV+uW/htC+t6M4dvKrUevjKrty4vq5pDSu72yb39bROof+fSH+f1F/KnbKyp83q624Z8SSH3D1Qal3U393S219qW9pfav1hX6l1Nf3/Kvp+I/EnxJuJv1vZ3XI3Pvt7Wu/v62l5sK/U8gDV2R/p2Mf6u4illieJf6W8mPTbiyt7Wl4aiaTrWPcGW9et7GldPxzpmE30uW31wrbXBxa23r2yNG328p7mT6B/iO6ioFA71q413rzCyu0Jpxvk5OJmf8/U/Vb3tHxioLvlC30Lcl8kR/8t6gyH9XW36mB/99SpK0u5FnIUx/b1th7f39vyPyt6Wk8j53Fu2hwoObW5Az2tF1H5L6YO/3/MnpZfE+8hh0/OLUdOIfc3Kvsb7G59EVxZavEY4EjiJJwVfb4MUvk30DW9IriZHNQW+v1Vup6tVNbX6Pdtq3rbBgeIqxdNH7xsydHMyy+cwbxi6bGD11x8/ODVo4BXLTtuB6+86Di+NuaFx+643rUXzBhcc/4xO/Eyn0tAr35+OISXLh5NnL4zhV1ct/zEwSupvkjgfrdqUdtxFESpRzFksdYw3rx06Yy3YkoAjnvpXGPvuIb+SxfM2HdgXtun4Jwp0v0eOWG9vyt3DPE0+vv0/q6pZ/T35M4mh3AOOYRZ5ADbg5xjFKTz5Pu7WzvoPOeRI+omZ7pooKflfPq+lM4/QH9TxJlDxPlL+t/vKXp8tNy5JciNwon+g1lq2Q6nic7hdxA4CziQtXCcFx3LTgbOJnUOdNnxXDbQd3w7nJ1wZj8chrs6iyT5hoPyCRHbwYVvfK5a2LaDaLed2TpIwqj4T0oEQJdTEAD7pv57xapFreMta+xbhOtOBxD1r7Wst2E+ebFl8Dxy/6Kp712xMPfpVQunH9zX2/a9Fd25yeQ4TxmgCJuc1Gkre1tOpws8s7+3dRY5MMt3vPS9RBHgYnJioUkOeQmxj/K5gs5xHX3eQv+/g3gvOfCH6fNJ+nyBnPhGOs9WdC46Zjt9f50+t4mo8x/0+Srls4Xy2Ezf/06ffpRaHSkPutataFDRkINXkYO79pITPC4/gSIY4ooTB/+XiE9ENPiffwwc9e4Y6EyHIznZ8ogTZSmPOv1o04ssg5xbAhziQMudZqDjpLoG/Y6kqKiYLqKfwvcguKP+fSuNYr8tJCV9GBzUxgxa1psgdpZlvQUjpt7ecXtA7Nb0tr5voPuYD/XNn/ZxzNVjuokE5Mskcl8jgRs7UGo7ikRtUn93biYJzCwShbk8sqiG3fxJgte6iMRp4cre1h4WTRqx9HW3dtH/FlDlzqffz+PjelpW0P8vo9ENCWLuxr6ell/RsfcRH6BjMSf+NJXnefpOotj6dwhetRRit4nKs5HOv4HO/TLlGTQf/TL9tpmO2QYjgIOHEVxJIxNflCBSOzFIyKIki6IQRiGIzLJpGH8KZmdhFJF+kiwXR4kRhRpVKCrWl+hrvs9Ye8ExV166pA1bjKXrzS2jFiTWvkCv6TqVVzlBqDHluKZr+j642V5vIihYvSD36dWltq+u6MqNI9HLkmFM7+uadtaKBbl2GhG306i3fUWpdU7fAjEFWcrZ5Jzrwr5SizvQk5vb153rou8LKRi5kAR4GYn2CirbQF+p9Rr6xM30O/pKud9QeR6kvxEgvETlXkfHJUoEFRxo4D6RGE1D4CB6/v0P6mjc4TBlgpGsL+IQ99Cj5LhIZfPL6U95+oEG6E1tBoh+mklt4nNHYFLOIUGKz6BABURbK6aHaB9/5EbteeeqxUfX/r5PBYW4gDl0BAj+lDXud8YVGOyOyxe0fBQLVlb2TDmQgoavUKDwXQoUjlxZaptII//cytK0aSu7czPA/u42zCacQmJ+FohAwhf7OInAAqRgJk/CzCSxLjBLuSI+B3paz8PUPLOn5QYS8pvo+8107P0UYOBe6rN0TKDwp5HkBNdzQMIzHS2b6H+bvRkN3DpoeQ3E7AYdt53+x/djfeHDogx/FgFi70+7Y+YhtUFKDYRQ+PSvFUSQ5o2OymZWOJAL4tE7L2yhoMhn5LMw1E4oH9pnzfkzrrrsgmOPSN09NwUFhZEBrIjF4qq1S2fsnYYgQ5a4h49HEX64pOWjK0q5j/WXph6Ae/rLu1o+t7yn5aDVvbkv4hZHX2/bt/u7Wsf2LWw9nES8ob+3ZQIFKHgkoZUCl5b+3txMChROGehpObmvNO2c1QFBxEhmXzcHOktwSwefJPirSPivJK6lwOA64m0UKNxKv/+cfruvr6flIeKDFDA8RH8/uoM9rU/0deee7i/lnsIn5fscBX3P4ZP4IuUXGIyEJfIhkVu3evHRN162ePoRwkwVFBQUFBTCw7KsN4GDg4NjbuiducdSa8Y7MLuydOmMd+D2i08sze+1Jr/LJwVH3jOY9Bnl7Rnk8+Nlp+4jiqegoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCQImja/wM2osq4hJkuEQAAAABJRU5ErkJggg=="""
BARRICK_LOGO_DATA_URI = f"data:image/png;base64,{BARRICK_LOGO_B64}"


def _barrick_logo_html(width_px: int = 118) -> str:
    """Return a compact embedded Barrick logo for page headers and report preview cards."""
    return f'<img src="{BARRICK_LOGO_DATA_URI}" alt="Barrick" style="width:{width_px}px; height:auto; display:block;" />'


def _safe_key(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in text)[:60]


def _default_index(options: list[str], candidates: list[str], fallback: int = 0) -> int:
    lowered = {option.casefold(): index for index, option in enumerate(options)}
    for candidate in candidates:
        if candidate.casefold() in lowered:
            return lowered[candidate.casefold()]
    return min(fallback, max(len(options) - 1, 0))


def _configured_max_grades() -> int:
    return max(9, int(display_defaults().get("max_grade_variables", 9)))


def _display_report_title(title: str | None, fallback: str = "Resource Tabulation") -> str:
    clean = str(title or "").strip()
    if clean == "BM_Template Evaluation":
        return "Block Model Evaluation"
    return clean or fallback


def _format_header_chip_text(value: Any, max_chars: int = 34) -> str:
    """Keep page-header chips compact while preserving the analytical context."""
    text = str(value or "").strip()
    if not text:
        return "N/A"
    return text if len(text) <= max_chars else text[: max_chars - 1].rstrip() + "…"


def _page_header_chips(title: str) -> list[tuple[str, str]]:
    """Build compact context chips for the executive dashboard page header."""
    chips: list[tuple[str, str]] = []
    models = st.session_state.get("models", {})
    if isinstance(models, dict):
        chips.append(("Models", f"{len(models):,}"))

    blk_scope = st.session_state.get("master_blk_model_scope")
    if blk_scope:
        chips.append(("BLK_MODEL", _format_header_chip_text(blk_scope, 22)))

    # Use the app helpers when available; fall back to raw session-state values during early page loads.
    try:
        year_label = _master_year_label()  # type: ignore[name-defined]
    except Exception:
        year_label = st.session_state.get("master_year_scope", "LOM")
    if year_label and title not in {"Model Setup"}:
        chips.append(("Year", _format_header_chip_text(year_label, 30)))

    try:
        phase_label = _master_phase_label()  # type: ignore[name-defined]
    except Exception:
        phase_label = "All available phases"
    if phase_label and title not in {"Model Setup"}:
        chips.append(("Phase", _format_header_chip_text(phase_label, 34)))

    try:
        destination_label = _master_destination_label()  # type: ignore[name-defined]
    except Exception:
        destination_label = st.session_state.get("master_destination_mode", "HG+LG")
    if destination_label and title not in {"Model Setup"}:
        chips.append(("Destination", _format_header_chip_text(destination_label, 28)))

    return chips


def page_header(title: str, subtitle: str | None = None) -> None:
    """Premium executive dashboard header used consistently across app pages."""
    chips = _page_header_chips(title)
    chip_html = "".join(
        f"""
        <div class=\"bm-page-header-chip\">
            <span class=\"bm-page-header-chip-label\">{html.escape(label)}</span>
            <span class=\"bm-page-header-chip-value\">{html.escape(value)}</span>
        </div>
        """
        for label, value in chips
    )
    subtitle_html = (
        f"<div class=\"bm-page-header-subtitle\">{html.escape(str(subtitle))}</div>"
        if subtitle
        else ""
    )
    logo_html = _barrick_logo_html(112)
    st.markdown(
        f"""
        <section class=\"bm-page-header-card\">
            <div class=\"bm-page-header-content\">
                <div>
                    <div class=\"bm-page-header-kicker\">PV BlockModel Studio</div>
                    <div class=\"bm-page-header-title\">{html.escape(str(title))}</div>
                    {subtitle_html}
                </div>
                <div class=\"bm-page-header-chips\">{chip_html}</div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def _apply_premium_tab_styles() -> None:
    st.markdown(
        """
        <style>
        div[data-testid="stTabs"] {
            margin-top: 0.65rem;
        }

        div[data-testid="stTabs"] div[role="tablist"] {
            gap: 0.45rem;
            padding: 0.35rem;
            border: 1px solid rgba(0, 84, 124, 0.16);
            border-radius: 12px;
            background:
                linear-gradient(180deg, rgba(0, 84, 124, 0.085), rgba(0, 84, 124, 0.025));
            box-shadow:
                inset 0 1px 0 rgba(255, 255, 255, 0.85),
                0 8px 22px rgba(0, 45, 67, 0.055);
        }

        div[data-testid="stTabs"] button[role="tab"] {
            min-height: 2.45rem;
            padding: 0.58rem 0.95rem !important;
            border: 1px solid transparent !important;
            border-radius: 9px !important;
            background: transparent !important;
            color: #344B5A !important;
            font-weight: 650 !important;
            letter-spacing: 0.01em;
            transition:
                background-color 140ms ease,
                border-color 140ms ease,
                box-shadow 140ms ease,
                color 140ms ease;
        }

        div[data-testid="stTabs"] button[role="tab"] p {
            margin: 0;
            color: inherit !important;
            font-size: 0.92rem;
            font-weight: 650;
        }

        div[data-testid="stTabs"] button[role="tab"]:hover {
            border-color: rgba(0, 84, 124, 0.18) !important;
            background: rgba(255, 255, 255, 0.72) !important;
            color: #004967 !important;
        }

        div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
            border-color: rgba(0, 73, 103, 0.62) !important;
            background: linear-gradient(135deg, #004967 0%, #006A93 100%) !important;
            color: #FFFFFF !important;
            box-shadow:
                0 9px 22px rgba(0, 73, 103, 0.20),
                inset 0 1px 0 rgba(255, 255, 255, 0.22);
        }

        div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] p {
            color: #FFFFFF !important;
        }

        div[data-testid="stTabs"] div[data-baseweb="tab-highlight"] {
            display: none;
        }

        div[data-testid="stTabs"] div[role="tabpanel"] {
            padding-top: 1.1rem;
        }

        @media (max-width: 760px) {
            div[data-testid="stTabs"] div[role="tablist"] {
                overflow-x: auto;
                flex-wrap: nowrap;
            }

            div[data-testid="stTabs"] button[role="tab"] {
                flex: 0 0 auto;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _column_name_tokens(column: str) -> tuple[str, str, list[str]]:
    normalized = re.sub(r"[^A-Z0-9]+", "_", str(column).upper()).strip("_")
    compact = normalized.replace("_", "")
    tokens = [token for token in normalized.split("_") if token]
    return normalized, compact, tokens


def _is_contained_metal_column(column: str) -> bool:
    normalized, compact, tokens = _column_name_tokens(column)
    lower = str(column).casefold()
    return (
        any(token in lower for token in ["oz", "ounce", "ounces", "contained"])
        or compact.endswith("OZ")
        or compact.endswith("OUNCES")
        or compact in {"AUOZ", "AGOZ", "CUOZ"}
        or any(token in tokens for token in ["OZ", "OUNCES", "CONTAINED"])
    )


def _is_grade_accumulator_column(column: str) -> bool:
    """Return True for grade x tonnes accumulator columns such as AUTON, CUTON, STON, CTON or AGTON."""
    normalized, compact, tokens = _column_name_tokens(column)
    grade_accumulator_bases = {
        "AU",
        "AUCN",
        "AG",
        "CU",
        "CUT",
        "CUCN",
        "ZN",
        "PB",
        "AS",
        "SB",
        "TE",
        "MO",
        "NI",
        "FE",
        "MN",
        "HG",
        "CO",
        "BI",
        "W",
        "C",
        "CTOT",
        "TC",
        "OC",
        "S",
        "STOT",
        "TS",
        "S2",
        "CAO",
        "SIO2",
        "AL2O3",
        "MGO",
        "K2O",
        "NA2O",
        "P2O5",
        "AULTP",
        "AGLTP",
        "CULTP",
        "CLTP",
        "SLTP",
        "S2LTP",
        "OCLTP",
    }

    accumulator_tokens = {"TON", "TONS", "TONNE", "TONNES", "XTOM", "XTON"}
    if any(token in accumulator_tokens for token in tokens):
        return any(token in grade_accumulator_bases for token in tokens)

    for suffix in ("TON", "TONS", "TONNE", "TONNES", "XTON"):
        if compact.endswith(suffix):
            base = compact[: -len(suffix)]
            return any(base == grade_base or base.startswith(grade_base) for grade_base in grade_accumulator_bases)

    return False


def _is_known_categorical_column_name(column: str) -> bool:
    normalized, compact, tokens = _column_name_tokens(column)

    exact = {
        "DOMAIN",
        "DOM",
        "DOM_S",
        "MINERAL_DOMAIN",
        "EST_DOMAIN",
        "ESTIMATION_DOMAIN",
        "CATEGORY",
        "CATEG",
        "CATEG_GC",
        "CAT",
        "CLASS",
        "RESOURCE_CATEGORY",
        "BENCH",
        "METTYPE",
        "MET_TYPE",
        "METALLURGICAL_TYPE",
        "DEST",
        "DESTINATION",
        "DESTINATION_LTP",
        "LITHO",
        "LITHOLOGY",
        "LITO",
        "LITOLOGIA",
        "ALTERATION",
        "ALTER",
        "ALT",
        "WEATHERING",
        "WEATH",
        "YEAR",
        "YEARS",
        "ANIO",
        "ANO",
        "PHASE",
        "PIT_PHASE",
        "PIT",
        "BLK_MODEL",
        "BLOCK_MODEL",
        "BM_TYPE",
        "SOURCE",
        "SRC",
        "REGION",
    }
    compact_exact = {value.replace("_", "") for value in exact}

    if normalized in exact or compact in compact_exact:
        return True

    if "LITH" in compact or "LITO" in compact:
        return True
    if "ALTER" in compact or compact.startswith("ALT"):
        return True
    if "WEATH" in compact or "WX" in tokens:
        return True
    if "DOMAIN" in compact:
        return True
    if "CATEG" in compact or compact.startswith("CAT"):
        return True
    if "DEST" in compact:
        return True

    return False


def _is_grade_like_column(column: str) -> bool:
    if (
        _is_contained_metal_column(column)
        or _is_grade_accumulator_column(column)
        or _is_known_categorical_column_name(column)
    ):
        return False

    configured = {str(name).casefold() for name in grade_defaults().get("grade_candidates", [])}
    if str(column).casefold() in configured:
        return True

    normalized, compact, tokens = _column_name_tokens(column)
    grade_tokens = {
        "AU",
        "AUCN",
        "AG",
        "CU",
        "CUT",
        "CUCN",
        "ZN",
        "PB",
        "AS",
        "SB",
        "TE",
        "MO",
        "NI",
        "FE",
        "MN",
        "HG",
        "CO",
        "BI",
        "W",
        "C",
        "CTOT",
        "TC",
        "OC",
        "S",
        "STOT",
        "TS",
        "S2",
        "CAO",
        "SIO2",
        "AL2O3",
        "MGO",
        "K2O",
        "NA2O",
        "P2O5",
    }
    compact_prefixes = (
        "AUCN",
        "AU",
        "AG",
        "CUCN",
        "CUT",
        "CU",
        "ZN",
        "PB",
        "AS",
        "SB",
        "TE",
        "STOT",
        "CTOT",
        "S2",
        "OC",
        "CAO",
        "SIO2",
        "AL2O3",
        "MGO",
        "K2O",
        "NA2O",
        "P2O5",
        "CLTP",
        "SLTP",
        "S2LTP",
        "OCLTP",
        "AULTP",
        "AGLTP",
        "CULTP",
    )

    if any(token in grade_tokens for token in tokens):
        return True

    if compact.startswith(compact_prefixes):
        return True

    return False


def _is_category_candidate(frame: pd.DataFrame, column: str) -> bool:
    if _is_setup_excluded_category_column(column) or _is_grade_like_column(column):
        return False

    if _is_known_categorical_column_name(column):
        return True

    series = frame[column]
    if pd.api.types.is_bool_dtype(series) or pd.api.types.is_categorical_dtype(series):
        return True
    if series.dtype == "object" or pd.api.types.is_string_dtype(series):
        return True

    non_null = int(series.notna().sum())
    unique_count = int(series.nunique(dropna=True))
    if non_null == 0:
        return False

    return unique_count <= 100 and unique_count <= max(30, int(non_null * 0.20))


def _default_grade_unit(column: str) -> str:
    normalized, compact, tokens = _column_name_tokens(column)

    for rule in grade_defaults().get("unit_rules", []):
        token = str(rule.get("contains", "")).casefold()
        if token and token in str(column).casefold().replace(" ", "_"):
            return str(rule.get("unit", "g/t"))

    if "PPM" in tokens or compact.endswith("PPM"):
        return "ppm"
    if any(token in tokens for token in ["GT", "GPT"]):
        return "g/t"
    if compact.startswith(("CU", "CUT", "CUCN", "CULTP", "CLTP", "OCLTP", "SLTP", "S2LTP", "STOT", "CTOT", "CAO", "SIO2", "AL2O3", "MGO", "K2O", "NA2O", "P2O5")):
        return "%"
    if any(token in tokens for token in ["CU", "CUT", "CUCN", "C", "OC", "S", "S2", "STOT", "CTOT", "CAO", "SIO2", "AL2O3"]):
        return "%"
    return "g/t"


def _default_grade_label(column: str) -> str:
    labels = grade_defaults().get("default_labels", {})
    direct = labels.get(column, labels.get(str(column).upper()))

    normalized, compact, tokens = _column_name_tokens(column)
    if direct:
        direct_text = str(direct).strip()
        if direct_text.casefold() == "s" and (any(token in tokens for token in ["STOT", "S"]) or compact.startswith("SLTP")):
            return "Stot"
        if direct_text.casefold() == "c" and (any(token in tokens for token in ["CTOT", "C"]) or compact.startswith("CLTP")):
            return "Ctot"
        return direct_text
    if compact.startswith("AUCN") or "AUCN" in tokens:
        return "AuCN"
    if "AU" in tokens or compact.startswith("AU"):
        return "Au"
    if "AG" in tokens or compact.startswith("AG"):
        return "Ag"
    if compact.startswith("CUCN") or "CUCN" in tokens:
        return "CuCN"
    if compact.startswith("CUT") or "CUT" in tokens:
        return "CuT"
    if "CU" in tokens or compact.startswith("CU"):
        return "Cu"
    if "CAO" in tokens or compact.startswith("CAO"):
        return "CaO"
    if "SIO2" in tokens or compact.startswith("SIO2"):
        return "SiO2"
    if "AL2O3" in tokens or compact.startswith("AL2O3"):
        return "Al2O3"
    if "OC" in tokens or compact.startswith("OC"):
        return "OC"
    if "S2" in tokens or compact.startswith("S2"):
        return "S2"
    if any(token in tokens for token in ["CTOT", "TC", "C"]) or compact.startswith("CLTP"):
        return "Ctot"
    if any(token in tokens for token in ["STOT", "TS", "S"]) or compact.startswith("SLTP"):
        return "Stot"
    return str(column)


def _infer_role(column: str) -> str:
    normalized, compact, tokens = _column_name_tokens(column)

    direct_aliases = {
        "Domain": {"DOMAIN", "DOM", "DOM_S", "MINERAL_DOMAIN", "EST_DOMAIN", "ESTIMATION_DOMAIN"},
        "Category": {"CATEGORY", "CATEG", "CATEG_GC", "CAT", "CLASS", "RESOURCE_CATEGORY"},
        "Bench": {"BENCH", "BENCH_ID", "RL", "ELEVATION"},
        "Mettype": {"METTYPE", "MET_TYPE", "METALLURGICAL_TYPE", "MET"},
        "Destination": {"DESTINATION", "DESTINATION_LTP", "DEST", "DEST_CODE"},
        "Lithology": {"LITHO", "LITHOLOGY", "LITO", "LITOLOGIA"},
        "Alteration": {"ALTERATION", "ALTER", "ALT"},
        "Weathering": {"WEATHERING", "WEATH", "WX"},
        "Year": {"YEAR", "YEARS", "ANIO", "ANO"},
        "Pit_Phase": {"PIT_PHASE", "PITPHASE"},
        "Phase": {"PHASE", "PHASE_ID"},
        "Pit": {"PIT", "PIT_NAME"},
        "Block Model": {"BLK_MODEL", "BLOCK_MODEL", "BMTYPE", "BM_TYPE"},
        "Source": {"SOURCE", "SRC"},
    }

    for role, aliases in direct_aliases.items():
        alias_compacts = {alias.replace("_", "") for alias in aliases}
        if normalized in aliases or compact in alias_compacts:
            return role if role in CATEGORY_ROLES else "Other"

    name = column.casefold().replace("_", " ").replace("-", " ")
    for rule in column_aliases().get("role_keywords", []):
        role = str(rule.get("role", "Other"))
        tokens_from_rule = [str(token).casefold() for token in rule.get("tokens", [])]
        if any(token in name for token in tokens_from_rule):
            return role if role in CATEGORY_ROLES else "Other"
    return "Other"


def _suggest_grades(columns: list[str]) -> list[str]:
    candidates = [column for column in columns if _is_grade_like_column(column)]

    preferred_labels = ["Au", "AuCN", "Ag", "Cu", "CuT", "CuCN", "Stot", "S2", "Ctot", "OC", "CaO", "SiO2", "Al2O3"]
    ordered: list[str] = []
    for label in preferred_labels:
        found = next((column for column in candidates if _default_grade_label(column).casefold() == label.casefold()), None)
        if found and found not in ordered:
            ordered.append(found)
    ordered.extend(column for column in candidates if column not in ordered)
    return ordered[:_configured_max_grades()]


def _suggest_categories(frame: pd.DataFrame) -> list[str]:
    preferred = column_aliases().get("category_priority", [])
    present = [column for column in preferred if column in frame.columns and _is_category_candidate(frame, column)]
    if present:
        return present[:10]

    known = [column for column in frame.columns if _is_known_categorical_column_name(column) and _is_category_candidate(frame, column)]
    fallback = [column for column in frame.columns if column not in known and _is_category_candidate(frame, column)]
    return (known + fallback)[:10]


def _is_setup_excluded_category_column(column: str) -> bool:
    normalized = str(column).strip().casefold().replace(" ", "_").replace("-", "_")
    return normalized in {"source", "src", "blk_model", "block_model", "bm_type"}


def _first_available_role(config: ModelConfig, roles: list[str]) -> str | None:
    for role in roles:
        column = config.column_for_role(role)
        if column:
            return column
    return config.category_columns[0] if config.category_columns else None


def _volume_display_divisor_and_label(unit: str | None) -> tuple[float, str]:
    """Return the divisor and display label used for volume summaries.

    The model files normally store block volume in cubic metres. The app displays
    volume in a larger unit, by default Mm3, to keep the figures readable.
    """
    normalized = str(unit or "Mm3").strip().casefold().replace("³", "3").replace("^", "")
    normalized = normalized.replace(" ", "")
    if normalized in {"mm3", "millionm3", "millionsm3", "millioncubicmetres", "millioncubicmeters"}:
        return 1_000_000.0, "Mm3"
    if normalized in {"km3", "cubic kilometre", "cubickilometre", "cubickilometer", "kilometres3", "kilometers3"}:
        return 1_000_000_000.0, "km3"
    if normalized in {"m3", "cubicmetre", "cubicmeter"}:
        return 1.0, "m3"
    if normalized in {"ft3", "cubicfeet", "cubicfoot"}:
        # Keep the original values when the user explicitly selects ft3.
        # Unit conversion from m3 to ft3 is intentionally not inferred here.
        return 1.0, "ft3"
    return 1_000_000.0, "Mm3"


def _display_volume_total(data: pd.DataFrame, config: ModelConfig) -> tuple[float, str]:
    divisor, label = _volume_display_divisor_and_label(config.volume_unit)
    return total_volume(data, config) / divisor, label


def _format_volume_value(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):,.0f}"


def _scale_volume_columns_for_display(table: pd.DataFrame, bundles: dict[str, ModelBundle] | None = None) -> pd.DataFrame:
    """Scale volume columns in generic validation/comparison tables for display only."""
    if table.empty:
        return table.copy()

    unit = None
    if bundles:
        units = {str(bundle.config.volume_unit or "Mm3") for bundle in bundles.values() if bundle.config.volume_column}
        unit = units.pop() if len(units) == 1 else "Mm3"
    divisor, label = _volume_display_divisor_and_label(unit or "Mm3")

    display = table.copy()
    rename: dict[str, str] = {}
    for column in display.columns:
        lower = str(column).casefold()
        if "volume" in lower and pd.api.types.is_numeric_dtype(display[column]):
            display[column] = display[column] / divisor
            if label.casefold() not in lower:
                rename[column] = f"{column} ({label})"
    if rename:
        display = display.rename(columns=rename)
    return display


def _render_dashboard_kpi_cards(cards: list[dict[str, str]]) -> None:
    card_html = "".join(
        "<div class=\"bm-kpi-card\" "
        f"style=\"background:{card.get('bg', '#E7E6E6')}; border-bottom-color:{card.get('border', '#C9C9C9')};\">"
        f"<div class=\"bm-kpi-value\">{html.escape(str(card.get('value', '')))}</div>"
        f"<div class=\"bm-kpi-label\">{html.escape(str(card.get('label', '')))}</div>"
        "</div>"
        for card in cards
    )
    st.markdown(
        f"""
        <style>
        .bm-kpi-grid {{
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 1.15rem;
            margin: 0.35rem 0 1.25rem 0;
        }}
        .bm-kpi-card {{
            min-height: 7.25rem;
            border-radius: 0;
            border: 1px solid rgba(0,0,0,0.04);
            border-bottom: 4px solid;
            padding: 1.05rem 0.95rem 0.85rem 0.95rem;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.45);
        }}
        .bm-kpi-value {{
            color: #20252B;
            font-size: clamp(1.85rem, 2.6vw, 2.55rem);
            line-height: 1.0;
            font-weight: 500;
            letter-spacing: -0.015em;
            text-align: center;
            word-break: break-word;
        }}
        .bm-kpi-label {{
            color: #5B5B5B;
            font-size: 0.92rem;
            line-height: 1.15;
            margin-top: 0.62rem;
            text-align: center;
            font-weight: 500;
        }}
        @media (max-width: 980px) {{
            .bm-kpi-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
        }}
        @media (max-width: 560px) {{
            .bm-kpi-grid {{ grid-template-columns: 1fr; }}
        }}
        </style>
        <div class=\"bm-kpi-grid\">{card_html}</div>
        """,
        unsafe_allow_html=True,
    )


def _metric_row(bundle: ModelBundle, data: pd.DataFrame) -> None:
    config = bundle.config
    tonnage_value = total_tonnage(data, config) / {"t": 1, "Kt": 1_000, "Mt": 1_000_000}.get(config.tonnage_unit, 1_000_000)
    if config.volume_column:
        display_volume, volume_label = _display_volume_total(data, config)
        volume_value = _format_volume_value(display_volume)
    else:
        volume_label = config.volume_unit
        volume_value = "N/A"
    issue_count = len(bundle.validation.issues)
    issue_bg = "#E6F4EA" if issue_count == 0 else "#FCE8E6"
    issue_border = "#70AD47" if issue_count == 0 else "#F16349"

    _render_dashboard_kpi_cards([
        {"label": "Rows", "value": f"{len(data):,}", "bg": "#D9E2F3", "border": "#AEBBD0"},
        {"label": f"Tonnage ({config.tonnage_unit})", "value": f"{tonnage_value:,.{config.tonnage_decimals}f}", "bg": "#E7E6E6", "border": "#C9C9C9"},
        {"label": f"Volume ({volume_label})" if config.volume_column else "Volume", "value": volume_value, "bg": "#EFE5A1", "border": "#D6C45A"},
        {"label": "Validation issues", "value": f"{issue_count:,}", "bg": issue_bg, "border": issue_border},
    ])


def _resource_metric_row(bundle: ModelBundle, data: pd.DataFrame) -> None:
    config = bundle.config
    tonnage_value = _metric_tonnage(data, config)
    error_count = int(bundle.validation.error_count)
    warning_count = int(bundle.validation.warning_count)

    error_bg = "#E6F4EA" if error_count == 0 else "#FCE8E6"
    error_border = "#70AD47" if error_count == 0 else "#F16349"
    warning_bg = "#E6F4EA" if warning_count == 0 else "#EFE5A1"
    warning_border = "#70AD47" if warning_count == 0 else "#D6C45A"

    _render_dashboard_kpi_cards([
        {"label": "Rows", "value": f"{len(data):,}", "bg": "#D9E2F3", "border": "#AEBBD0"},
        {"label": f"Tonnes ({config.tonnage_unit})", "value": f"{tonnage_value:,.{config.tonnage_decimals}f}", "bg": "#E7E6E6", "border": "#C9C9C9"},
        {"label": "Validation errors", "value": f"{error_count:,}", "bg": error_bg, "border": error_border},
        {"label": "Warnings", "value": f"{warning_count:,}", "bg": warning_bg, "border": warning_border},
    ])

def _filter_panel(bundle: ModelBundle, key_prefix: str) -> tuple[pd.DataFrame, dict[str, list[Any]]]:
    data = bundle.data
    config = bundle.config
    selections: dict[str, list[Any]] = {}

    with st.expander("Filters", expanded=False):
        filter_columns = st.multiselect(
            "Active filters",
            options=config.category_columns,
            default=[column for column in config.category_columns if config.column_for_role("Destination") == column or config.column_for_role("Year") == column][:2],
            format_func=lambda col: f"{config.category_label(col)} [{col}]",
            key=f"{key_prefix}_active_filters",
        )
        for column in filter_columns:
            values = sorted(data[column].dropna().unique().tolist())
            if not values:
                continue
            selections[column] = st.multiselect(
                config.category_label(column),
                options=values,
                default=[],
                key=f"{key_prefix}_filter_{_safe_key(column)}",
            )

    return apply_categorical_filters(data, selections), selections


def _add_scene_button(title: str, kind: str, model_names: list[str], table: pd.DataFrame, filters: dict[str, Any] | None = None) -> None:
    """Report Builder is now automatic; manual scene buttons are intentionally disabled."""
    return


def _plot_tonnage_bar(table: pd.DataFrame, x_col: str, ton_col: str, color_col: str | None = None, title: str = "Tonnage") -> None:
    if table.empty or x_col not in table.columns or ton_col not in table.columns:
        return
    fig = px.bar(table, x=x_col, y=ton_col, color=color_col, title=title, color_discrete_sequence=CATEGORY_COLORS)
    fig.update_layout(height=430, legend_title_text=color_col)
    st.plotly_chart(fig, use_container_width=True)


METTYPE_COLORS = {
    # Corporate Block Model Mettype palette.
    # The Mettype plotting helpers convert values to uppercase before plotting.
    "NONE": "#FFFFFF",
    "MNSP": "#00FF00",
    "MOVC": "#00C8C8",
    "DEF": "#0057D9",
    "MOBS": "#666666",
    "MNVC": "#A6A6A6",
    "MNBS": "#BFBFBF",
    # Legacy aliases kept for compatibility with older model files.
    "VCL": "#00FF00",
    "VOLCANIC": "#00FF00",
    "BSD": "#666666",
    "BLACK SEDIMENT": "#666666",
}

RESOURCE_CATEGORY_COLORS = {
    "15_Grade Control": "#ED7D31",
    "0_Grade Control": "#ED7D31",
    "Grade Control": "#ED7D31",
    "1_Measured": "#5B9BD5",
    "Measured": "#5B9BD5",
    "2_Indicated": "#70AD47",
    "Indicated": "#70AD47",
    "3_Inferred": "#7E57C2",
    "Inferred": "#7E57C2",
    "4_Inventory": "#A5A5A5",
    "Inventory": "#A5A5A5",
}
RESOURCE_CATEGORY_ORDER = ["Grade Control", "Measured", "Indicated", "Inferred", "Inventory"]

METAL_AT_RISK_COLORS = {
    "Grade Control": "#44546A",
    "Measured": "#A39161",
    "Indicated (At Risk)": "#BFBFBF",
}

DESTINATION_COLORS = {
    "H1": "#B30000",
    "H2": "#FF0000",
    "L1": "#0033CC",
    "L2": "#0000FF",
    "L3": "#4F81BD",
    "M1": "#9E480E",
    "M2": "#C55A11",
    "M3": "#F4B183",
    "MW1": "#7030A0",
    "MW2": "#A64DFF",
    "W1": "#4D4D4D",
    "W2": "#808080",
    "HG": "#FF0000",
    "LG": "#0000FF",
}

_master_defaults = master_filter_defaults()
VALID_DESTINATIONS = {str(value).casefold() for value in _master_defaults.get("valid_destinations", ["h1", "h2", "l1", "l2", "l3", "m1", "m2", "m3", "mw1", "mw2", "w1", "w2"])}
HG_DESTINATIONS = {str(value).casefold() for value in _master_defaults.get("hg_destinations", ["h1", "h2"])}
LG_DESTINATIONS = {str(value).casefold() for value in _master_defaults.get("lg_destinations", ["l1", "l2", "l3"])}
M_DESTINATIONS = {"m1", "m2", "m3"}
MW_DESTINATIONS = {"mw1", "mw2"}
W_DESTINATIONS = {"w1", "w2"}
VALID_DESTINATIONS = VALID_DESTINATIONS | W_DESTINATIONS
DESTINATION_MODE_OPTIONS = [
    "HG+LG",
    "HG+LG+MW",
    "HG",
    "LG",
    "M1",
    "M2",
    "M3",
    "M1+M2+M3",
    "MW1",
    "MW2",
    "MW1+MW2 (Mineral Waste)",
    "W1",
    "W2",
    "W1+W2 (Waste)",
    "All destinations",
]
DESTINATION_MODE_CODES = {
    "HG+LG": HG_DESTINATIONS | LG_DESTINATIONS,
    "HG+LG+MW": HG_DESTINATIONS | LG_DESTINATIONS | MW_DESTINATIONS,
    "HG": HG_DESTINATIONS,
    "HG only": HG_DESTINATIONS,
    "LG": LG_DESTINATIONS,
    "LG only": LG_DESTINATIONS,
    "M1": {"m1"},
    "M2": {"m2"},
    "M3": {"m3"},
    "M1+M2+M3": M_DESTINATIONS,
    "MW1": {"mw1"},
    "MW2": {"mw2"},
    "MW1+MW2 (Mineral Waste)": MW_DESTINATIONS,
    "W1": {"w1"},
    "W2": {"w2"},
    "W1+W2 (Waste)": W_DESTINATIONS,
    "All destinations": VALID_DESTINATIONS,
}

def _normalize_label(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().replace("_", " ").replace("-", " ")
    # Remove common numeric prefixes such as "1. Measured", "2_Indicated" or "3-Inferred".
    text = re.sub(r"^\s*\d+(?:\.0)?\s*[\.\-_:]?\s*", "", text).strip()
    return " ".join(text.split()).casefold()



def _normalize_destination_code(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().casefold().replace(" ", "").replace("_", "").replace("-", "")
    return text


def _display_destination(value: Any) -> str:
    code = _normalize_destination_code(value)
    return code.upper() if code else ""


def _block_model_column(config: ModelConfig, data: pd.DataFrame) -> str | None:
    configured = config.column_for_role("Block Model")
    if configured and configured in data.columns:
        return configured
    for column in data.columns:
        if str(column).strip().casefold() in {"blk_model", "block_model", "block model", "bm_type"}:
            return column
    return None


def _year_column(config: ModelConfig, data: pd.DataFrame) -> str | None:
    configured = config.column_for_role("Year")
    if configured and configured in data.columns:
        return configured
    for column in data.columns:
        if str(column).strip().casefold() in {"year", "years", "anio", "ano"}:
            return column
    return None


def _phase_column(config: ModelConfig, data: pd.DataFrame) -> str | None:
    """Return one unified mining-phase column, accepting either Phase or Pit_Phase roles."""
    for role in ["Phase", "Pit_Phase"]:
        configured = config.column_for_role(role)
        if configured and configured in data.columns:
            return configured

    aliases = {"phase", "phase_id", "pit_phase", "pitphase", "mining_phase", "mine_phase"}
    for column in data.columns:
        normalized = re.sub(r"[^a-z0-9]+", "_", str(column).casefold()).strip("_")
        compact = normalized.replace("_", "")
        if normalized in aliases or compact in {alias.replace("_", "") for alias in aliases}:
            return column
    return None


def _available_years_for_bundles(bundles: list[ModelBundle]) -> list[int]:
    years: set[int] = set()
    for bundle in bundles:
        column = _year_column(bundle.config, bundle.data)
        if not column:
            continue
        numeric = pd.to_numeric(bundle.data[column], errors="coerce")
        years.update(int(value) for value in numeric.dropna().unique() if float(value).is_integer())
    return sorted(years)


def _available_year_bounds(available_years: list[int]) -> tuple[int, int] | None:
    if not available_years:
        return None
    return min(available_years), max(available_years)


def _years_from_master_scope(available_years: list[int]) -> list[int]:
    if not available_years:
        return []

    mode = st.session_state.get("master_year_scope", "LOM")
    if mode == "All years":
        return available_years
    if mode == "Custom years":
        custom = st.session_state.get("master_custom_years", available_years)
        selected = [int(year) for year in custom if int(year) in available_years]
        return selected or available_years

    if mode == "LOM":
        start_year = available_years[0]
        end_year = available_years[-1]
    elif mode == "Single year":
        start_year = int(st.session_state.get("master_year_start", available_years[0]))
        end_year = start_year
    else:
        period_years = int(str(mode).removesuffix("Y"))
        start_year = int(st.session_state.get("master_year_start", available_years[0]))
        end_year = start_year + period_years - 1

    return [year for year in available_years if start_year <= year <= end_year]


def _master_year_label() -> str:
    mode = st.session_state.get("master_year_scope", "LOM")
    selected = st.session_state.get("master_selected_years", [])
    if not selected:
        return "All available years"
    if mode == "Custom years":
        return ", ".join(map(str, selected))
    return f"{mode}: {min(selected)}-{max(selected)}" if len(selected) > 1 else f"{mode}: {selected[0]}"


def _render_master_year_filter_sidebar(bundles: list[ModelBundle], key_prefix: str) -> list[int]:
    available_years = _available_years_for_bundles(bundles)
    if not available_years:
        st.session_state.master_selected_years = []
        return []
    year_bounds = _available_year_bounds(available_years)
    lom_help = (
        f"LOM uses the loaded data range: {year_bounds[0]}-{year_bounds[1]}."
        if year_bounds
        else "LOM uses the loaded data range."
    )

    st.sidebar.markdown(
        """
        <div class="bm-side-banner">
            <div class="bm-banner-kicker">Time Scope</div>
            <div class="bm-banner-title">Year filter</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    modes = ["LOM", "Single year", "2Y", "5Y", "10Y", "Custom years", "All years"]
    current_mode = st.session_state.get("master_year_scope", "LOM")
    if current_mode not in modes:
        current_mode = "LOM"
    mode = st.sidebar.selectbox(
        "Year scope",
        modes,
        index=modes.index(current_mode),
        key=f"{key_prefix}_master_year_scope",
        help=f"{lom_help} This scope is applied across evaluation, comparison and reports.",
    )
    st.session_state.master_year_scope = mode

    if mode in {"Single year", "2Y", "5Y", "10Y"}:
        current_start = int(st.session_state.get("master_year_start", available_years[0]))
        if current_start not in available_years:
            current_start = available_years[0]
        start = st.sidebar.selectbox(
            "Start year",
            available_years,
            index=available_years.index(current_start),
            key=f"{key_prefix}_master_year_start",
        )
        st.session_state.master_year_start = int(start)
    elif mode == "Custom years":
        current_custom = [
            int(year) for year in st.session_state.get("master_custom_years", available_years)
            if int(year) in available_years
        ]
        custom = st.sidebar.multiselect(
            "Years",
            available_years,
            default=current_custom or available_years,
            key=f"{key_prefix}_master_custom_years",
        )
        st.session_state.master_custom_years = [int(year) for year in custom]

    selected_years = _years_from_master_scope(available_years)
    st.session_state.master_selected_years = selected_years
    st.sidebar.caption(f"Years applied: **{', '.join(map(str, selected_years)) if selected_years else 'None'}**")
    return selected_years


def _natural_phase_sort_key(value: Any) -> tuple[Any, ...]:
    text = str(value).strip()
    parts = re.split(r"(\d+)", text)
    key: list[Any] = []
    for part in parts:
        if part.isdigit():
            key.append(int(part))
        else:
            key.append(part.casefold())
    return tuple(key)


def _available_phases_for_bundles(bundles: list[ModelBundle]) -> list[str]:
    phases: set[str] = set()
    for bundle in bundles:
        column = _phase_column(bundle.config, bundle.data)
        if not column or column not in bundle.data.columns:
            continue
        values = bundle.data[column].dropna().astype(str).str.strip()
        phases.update(value for value in values.tolist() if value)
    return sorted(phases, key=_natural_phase_sort_key)


def _master_phase_label() -> str:
    selected = [str(value) for value in st.session_state.get("master_selected_phases", []) if str(value).strip()]
    available = [str(value) for value in st.session_state.get("master_available_phases", []) if str(value).strip()]
    if not selected:
        return "All available phases"
    if available and set(selected) == set(available):
        return "All available phases"
    return _format_filter_note_value(selected, max_items=12)


def _render_master_phase_filter_sidebar(bundles: list[ModelBundle], key_prefix: str) -> list[str]:
    available_phases = _available_phases_for_bundles(bundles)
    st.session_state.master_available_phases = available_phases
    if not available_phases:
        st.session_state.master_selected_phases = []
        return []

    st.sidebar.markdown(
        """
        <div class="bm-side-banner">
            <div class="bm-banner-kicker">Mining Scope</div>
            <div class="bm-banner-title">Phase filter</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    current = [
        str(value) for value in st.session_state.get("master_selected_phases", available_phases)
        if str(value) in available_phases
    ]
    selected = st.sidebar.multiselect(
        "Phase / Pit phase",
        available_phases,
        default=current or available_phases,
        key=f"{key_prefix}_master_phase_filter",
        help="Transversal filter. Uses Phase or Pit_Phase as one unified mining-phase field across the app.",
    )
    st.session_state.master_selected_phases = selected or available_phases
    st.sidebar.caption(f"Phases applied: **{_master_phase_label()}**")
    return st.session_state.master_selected_phases


def _master_destination_label() -> str:
    mode = str(st.session_state.get("master_destination_mode", "HG+LG"))
    return mode if mode in DESTINATION_MODE_OPTIONS else "HG+LG"


def _render_master_destination_filter_sidebar(bundles: list[ModelBundle], key_prefix: str) -> str:
    has_destination = any(
        bundle.config.column_for_role("Destination") in bundle.data.columns
        for bundle in bundles
        if bundle.config.column_for_role("Destination")
    )
    if not has_destination:
        st.session_state.master_destination_mode = "All destinations"
        return "All destinations"

    st.sidebar.markdown(
        """
        <div class="bm-side-banner">
            <div class="bm-banner-kicker">Material Scope</div>
            <div class="bm-banner-title">Destination / Ore Type</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    current_mode = _master_destination_label()
    widget_key = f"{key_prefix}_master_destination_mode"
    if st.session_state.get(widget_key) != current_mode:
        st.session_state[widget_key] = current_mode

    mode = st.sidebar.selectbox(
        "Destination / Ore Type",
        DESTINATION_MODE_OPTIONS,
        index=DESTINATION_MODE_OPTIONS.index(current_mode),
        key=widget_key,
        help=(
            "Transversal master filter applied across Model Description, Model Evaluation, "
            "Model Comparison and Report Builder. HG+LG = H1/H2 + L1/L2/L3; HG+LG+MW adds "
            "MW1/MW2; All destinations includes all valid H/L/M/MW/W destinations."
        ),
    )
    st.session_state.master_destination_mode = mode
    st.sidebar.caption(f"Destination / Ore Type applied: **{mode}**")
    return mode


def _valid_destination_mask(data: pd.DataFrame, config: ModelConfig) -> pd.Series:
    dest_col = config.column_for_role("Destination")
    if not dest_col or dest_col not in data.columns:
        return pd.Series(True, index=data.index)
    normalized = data[dest_col].map(_normalize_destination_code)
    return normalized.isin(VALID_DESTINATIONS)


def _apply_global_scope(data: pd.DataFrame, config: ModelConfig) -> pd.DataFrame:
    """Apply transverse app rules before any calculations, tables or plots."""
    if data.empty:
        return data.copy()

    filtered = data.copy()

    blk_col = _block_model_column(config, filtered)
    if blk_col and blk_col in filtered.columns:
        valid_blk_values = selected_master_blk_model_values()
        blk_numeric = pd.to_numeric(filtered[blk_col], errors="coerce")
        filtered = filtered[blk_numeric.isin(valid_blk_values)]

    dest_col = config.column_for_role("Destination")
    if dest_col and dest_col in filtered.columns:
        filtered = filtered[_valid_destination_mask(filtered, config)]
        filtered = _apply_destination_mode(filtered, config, _master_destination_label())

    year_col = _year_column(config, filtered)
    if year_col and "master_selected_years" in st.session_state:
        selected_years = st.session_state.get("master_selected_years", [])
        years = pd.to_numeric(filtered[year_col], errors="coerce")
        filtered = filtered[years.isin([int(year) for year in selected_years])]

    phase_col = _phase_column(config, filtered)
    if phase_col and phase_col in filtered.columns and "master_selected_phases" in st.session_state:
        selected_phases = [str(value) for value in st.session_state.get("master_selected_phases", [])]
        if selected_phases:
            phases = filtered[phase_col].astype(str).str.strip()
            filtered = filtered[phases.isin(selected_phases)]

    return filtered


def _scoped_bundle(bundle: ModelBundle) -> ModelBundle:
    data = _apply_global_scope(bundle.data, bundle.config)
    validation = validate_model(data, bundle.config)
    return replace(bundle, data=data, validation=validation)


def _scope_caption(data_before: pd.DataFrame, data_after: pd.DataFrame, config: ModelConfig) -> None:
    blk_col = _block_model_column(config, data_before)
    dest_col = config.column_for_role("Destination")
    parts = [f"Rows after master scope: **{len(data_after):,}** of {len(data_before):,}"]
    if blk_col:
        parts.append(f"BLK_MODEL: **{st.session_state.get('master_blk_model_scope', '1 - In situ')}**")
    year_col = _year_column(config, data_before)
    if year_col:
        parts.append(f"Years: **{_master_year_label()}**")
    phase_col = _phase_column(config, data_before)
    if phase_col:
        parts.append(f"Phases: **{_master_phase_label()}**")
    if dest_col:
        parts.append(f"Destination / Ore Type: **{_master_destination_label()}**; CUT/NONE/invalid excluded")
    st.caption(" | ".join(parts))


def _format_filter_note_value(value: Any, max_items: int = 10) -> str:
    if value is None:
        return "Not applied"
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list):
        if len(value) == 2 and all(isinstance(item, (int, float)) for item in value):
            return f"{value[0]:g} to {value[1]:g}"
        if not value:
            return "No values selected"
        clean_values = [str(item) for item in value]
        if len(clean_values) > max_items:
            return ", ".join(clean_values[:max_items]) + f", +{len(clean_values) - max_items} more"
        return ", ".join(clean_values)
    if isinstance(value, set):
        return _format_filter_note_value(sorted(value), max_items=max_items)
    return str(value)


def _table_filter_note_items(
    config: ModelConfig,
    source_data: pd.DataFrame | None = None,
    final_data: pd.DataFrame | None = None,
    filters: dict[str, Any] | None = None,
    model_name: str | None = None,
    extra_items: list[str] | None = None,
) -> list[str]:
    data_for_roles = source_data if source_data is not None else final_data
    items: list[str] = []

    if model_name:
        items.append(f"Model: {model_name}")
    if source_data is not None and final_data is not None:
        items.append(f"Rows used in table: {len(final_data):,} of {len(source_data):,} after the selected scope/filters")
    elif final_data is not None:
        items.append(f"Rows used in table: {len(final_data):,}")

    if data_for_roles is not None:
        blk_col = _block_model_column(config, data_for_roles)
        if blk_col:
            items.append(f"Master BLK_MODEL scope: {st.session_state.get('master_blk_model_scope', '1 - In situ')}")

        year_col = _year_column(config, data_for_roles)
        if year_col:
            items.append(f"Master Year scope: {_master_year_label()}")

        phase_col = _phase_column(config, data_for_roles)
        if phase_col:
            items.append(f"Master Phase/Pit_Phase scope: {_master_phase_label()}")

        dest_col = config.column_for_role("Destination")
        if dest_col and dest_col in data_for_roles.columns:
            items.append(
                f"Master Destination / Ore Type scope: {_master_destination_label()}; "
                "CUT/NONE/invalid values excluded"
            )

    for key, value in (filters or {}).items():
        label = config.category_label(key) if isinstance(key, str) and key in config.category_columns else str(key)
        items.append(f"{label}: {_format_filter_note_value(value)}")

    if extra_items:
        items.extend(extra_items)
    return items


def _render_table_filter_note(items: list[str] | None) -> None:
    if not items:
        return

    # Keep footnotes only under resource-reporting tables.
    # Configuration, validation, audit and auxiliary filter/check tables intentionally stay clean.
    resource_note_markers = (
        "Resource-category table grouping",
        "Destination table grouping",
        "Comparison table grouping",
    )
    if not any(any(marker in str(item) for marker in resource_note_markers) for item in items):
        return

    html_items = "".join(f"<li>{html.escape(str(item))}</li>" for item in items if str(item).strip())
    if not html_items:
        return
    st.markdown(
        f"""
        <div style="font-size:0.72rem; color:#5B6573; margin:0.25rem 0 0.85rem 0; line-height:1.28;">
            <div style="font-weight:700; color:#344B5A; margin-bottom:0.10rem;">Filters applied to this resource table</div>
            <ol style="margin:0; padding-left:1.25rem;">{html_items}</ol>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _comparison_filter_note_items(
    bundles: dict[str, ModelBundle],
    selected_model_names: list[str],
    reference_model: str | None = None,
    extra_items: list[str] | None = None,
) -> list[str]:
    items = [f"Models compared: {', '.join(selected_model_names)}"]
    if bundles:
        row_counts = [f"{name}: {len(bundle.data):,} rows" for name, bundle in bundles.items()]
        items.append("Rows after master scope: " + "; ".join(row_counts))
        if any(_block_model_column(bundle.config, bundle.data) for bundle in bundles.values()):
            items.append(f"Master BLK_MODEL scope: {st.session_state.get('master_blk_model_scope', '1 - In situ')}")
        if any(_year_column(bundle.config, bundle.data) for bundle in bundles.values()):
            items.append(f"Master Year scope: {_master_year_label()}")
        if any(_phase_column(bundle.config, bundle.data) for bundle in bundles.values()):
            items.append(f"Master Phase/Pit_Phase scope: {_master_phase_label()}")
        if any(bundle.config.column_for_role("Destination") for bundle in bundles.values()):
            items.append(
                f"Master Destination / Ore Type scope: {_master_destination_label()}; "
                "CUT/NONE/invalid values excluded"
            )
    if reference_model:
        items.append(f"Reference model for Δ%: {reference_model}")
    if extra_items:
        items.extend(extra_items)
    return items


def _configured_models_inventory() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name, bundle in st.session_state.models.items():
        scoped = _scoped_bundle(bundle)
        rows.append(
            {
                "Model": name,
                "Type": bundle.config.model_type,
                "Rows after scope": len(scoped.data),
                "Raw rows": len(bundle.data),
                "Grades": len(bundle.config.grade_specs),
                "Categories": len(bundle.config.category_specs),
                "Validation": "Blocked" if scoped.validation.is_blocked else "Ready",
            }
        )
    return pd.DataFrame(rows)


def _render_configured_models_manager(key_prefix: str = "setup") -> None:
    st.markdown("### Configured Models")
    if not st.session_state.models:
        st.info("No models have been configured yet.")
        return

    st.dataframe(_configured_models_inventory(), use_container_width=True, hide_index=True)

    st.markdown("#### Delete model")
    st.caption("Use this control only when you need to remove a configured model before continuing with evaluation or comparison.")
    delete_cols = st.columns([2.2, 1.0, 1.0])
    model_to_delete = delete_cols[0].selectbox(
        "Model to delete",
        list(st.session_state.models),
        key=f"{key_prefix}_model_to_delete",
    )
    confirm_delete = delete_cols[1].checkbox(
        "Confirm delete",
        key=f"{key_prefix}_confirm_model_delete",
    )
    if delete_cols[2].button(
        "Delete",
        key=f"{key_prefix}_delete_model_button",
        disabled=not confirm_delete,
        use_container_width=True,
    ):
        st.session_state.models.pop(model_to_delete, None)
        st.session_state.pop(f"{key_prefix}_model_to_delete", None)
        st.session_state.pop(f"{key_prefix}_confirm_model_delete", None)
        st.success(f"Model '{model_to_delete}' deleted.")
        st.rerun()

def _display_resource_category(value: Any) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.notna(numeric):
        if numeric in {0, 15}:
            return "Grade Control"
        if numeric == 1:
            return "Measured"
        if numeric == 2:
            return "Indicated"
        if numeric == 3:
            return "Inferred"
        if numeric == 4:
            return "Inventory"
        if numeric == 5:
            return "Unclassified"

    norm = _normalize_label(value)
    if "grade" in norm and "control" in norm:
        return "Grade Control"
    if "measured" in norm:
        return "Measured"
    if "indicated" in norm:
        return "Indicated"
    if "inferred" in norm:
        return "Inferred"
    if "inventory" in norm:
        return "Inventory"
    if "unclassified" in norm or "unclass" in norm:
        return "Unclassified"
    return str(value) if pd.notna(value) else "Unclassified"

def _category_mask(data: pd.DataFrame, column: str | None, category_key: str) -> pd.Series:
    if not column or column not in data.columns:
        return pd.Series(False, index=data.index)

    raw = data[column]
    normalized = raw.map(_normalize_label)
    numeric = pd.to_numeric(raw, errors="coerce")

    if category_key == "grade_control":
        return numeric.isin([0, 15]) | (normalized.str.contains("grade", na=False) & normalized.str.contains("control", na=False))
    if category_key == "measured":
        return (numeric == 1) | normalized.str.contains("measured", na=False)
    if category_key == "indicated":
        return (numeric == 2) | normalized.str.contains("indicated", na=False)
    if category_key == "inferred":
        return (numeric == 3) | normalized.str.contains("inferred", na=False)
    if category_key == "inventory":
        return (numeric == 4) | normalized.str.contains("inventory", na=False)
    if category_key == "unclassified":
        return (numeric == 5) | normalized.str.contains("unclassified", na=False) | normalized.str.contains("unclass", na=False)
    return pd.Series(False, index=data.index)


def _is_grade_control_category_column(column: str | None) -> bool:
    if not column:
        return False
    normalized = re.sub(r"[^a-z0-9]+", "_", str(column).casefold()).strip("_")
    compact = normalized.replace("_", "")
    return normalized in {"categ_gc", "category_gc", "resource_category_gc"} or compact in {"categgc", "categorygc"}


def _is_contained_metal_spec(spec: GradeSpec) -> bool:
    text = f"{spec.column} {spec.label}".casefold()
    return any(token in text for token in ["oz", "ounce", "ounces", "contained"])


def _grade_name_tokens(spec: GradeSpec) -> tuple[str, str]:
    raw_label = str(spec.label or spec.column).strip()
    raw_column = str(spec.column).strip()
    normalized = re.sub(r"[^A-Z0-9]+", "_", f"{raw_label}_{raw_column}".upper()).strip("_")
    compact = normalized.replace("_", "")
    return normalized, compact


def _canonical_grade_label(spec: GradeSpec) -> str:
    normalized, compact = _grade_name_tokens(spec)
    tokens = [token for token in normalized.split("_") if token]

    if any(token in tokens for token in ["AUCN"]) or compact.startswith("AUCN") or compact.startswith("AUCNLTP"):
        return "AuCN"
    if any(token in tokens for token in ["AU", "GOLD"]):
        return "Au"
    if any(token in tokens for token in ["AG", "SILVER"]):
        return "Ag"
    if any(token in tokens for token in ["CUCN"]) or compact.startswith("CUCN"):
        return "CuCN"
    if any(token in tokens for token in ["CUT"]) or compact.startswith("CUT"):
        return "CuT"
    if any(token in tokens for token in ["CU", "COPPER"]):
        return "Cu"
    if any(token in tokens for token in ["CAO", "CALCIUMOXIDE"]):
        return "CaO"
    if any(token in tokens for token in ["SIO2", "SILICA", "SILICONOXIDE"]):
        return "SiO2"
    if any(token in tokens for token in ["AL2O3", "ALUMINA", "ALUMINUMOXIDE"]):
        return "Al2O3"
    if any(token in tokens for token in ["OC", "ORGC", "ORGANICC", "ORGANICCARBON"]):
        return "OC"
    if any(token in tokens for token in ["S2", "S2PCT", "S2LTP"]):
        return "S2"
    if any(token in tokens for token in ["CTOT", "TC", "CARBON", "C"]):
        return "Ctot"
    if any(token in tokens for token in ["STOT", "TS", "SULFUR", "S"]):
        return "Stot"

    # Fallbacks for compact names such as AULTP, AGLTP, CULTP, STOT or CTOT.
    if compact.startswith("AUCN"):
        return "AuCN"
    if compact.startswith("AU"):
        return "Au"
    if compact.startswith("AG"):
        return "Ag"
    if compact.startswith("CUCN"):
        return "CuCN"
    if compact.startswith("CUT"):
        return "CuT"
    if compact.startswith("CU"):
        return "Cu"
    if compact.startswith("CAO"):
        return "CaO"
    if compact.startswith("SIO2"):
        return "SiO2"
    if compact.startswith("AL2O3"):
        return "Al2O3"
    if compact.startswith("OC"):
        return "OC"
    if compact.startswith("S2"):
        return "S2"
    if compact.startswith("CTOT") or compact.startswith("CLTP") or compact == "C":
        return "Ctot"
    if compact.startswith("STOT") or compact.startswith("SLTP") or compact == "S":
        return "Stot"
    return str(spec.label or spec.column).strip()


def _effective_grade_unit(spec: GradeSpec) -> str:
    label = _canonical_grade_label(spec)
    normalized, compact = _grade_name_tokens(spec)
    configured_unit = str(spec.unit or "").strip()

    if configured_unit == "%":
        return "%"
    if "PPM" in normalized.split("_") or compact.endswith("PPM"):
        return "ppm"
    if any(token in normalized.split("_") for token in ["GT", "GPT", "G/T"]):
        return "g/t"

    # Pueblo Viejo LTP geochemistry commonly stores base/carbon/sulfur variables as percent.
    if label in {"Cu", "CuT", "CuCN", "Ctot", "OC", "Stot", "S2", "CaO", "SiO2", "Al2O3"}:
        return "%"
    if label in {"Au", "AuCN", "Ag"}:
        return "g/t"
    return configured_unit or "g/t"


def _resource_grade_header(spec: GradeSpec) -> str:
    label = _canonical_grade_label(spec)
    unit = _effective_grade_unit(spec)
    return f"{label} ({unit})"


def _preferred_grade_specs(config: ModelConfig) -> list[GradeSpec]:
    preferred = ["Au", "AuCN", "Ag", "Cu", "CuT", "CuCN", "Stot", "S2", "Ctot", "OC", "CaO", "SiO2", "Al2O3"]
    available = [spec for spec in config.grade_specs if not _is_contained_metal_spec(spec)]
    specs: list[GradeSpec] = []

    for label in preferred:
        found = next((spec for spec in available if _canonical_grade_label(spec).casefold() == label.casefold()), None)
        if found and found not in specs:
            specs.append(found)

    for spec in available:
        if spec not in specs and _canonical_grade_label(spec) not in preferred:
            specs.append(spec)
    return specs[:_configured_max_grades()]


def _find_metal_grade_spec(specs: list[GradeSpec], metal_label: str) -> GradeSpec | None:
    return next((spec for spec in specs if _canonical_grade_label(spec).casefold() == metal_label.casefold()), None)


def _resource_row(label: str, subset: pd.DataFrame, config: ModelConfig, specs: list[GradeSpec]) -> dict[str, Any]:
    ton_col = f"Tonnes ({config.tonnage_unit})"
    mass = subset[config.mass_column].clip(lower=0) if config.mass_column in subset.columns else pd.Series(dtype=float)
    row: dict[str, Any] = {"Category": label, ton_col: float(mass.sum()) / tonnage_divisor(config.tonnage_unit)}

    for spec in specs:
        header = _resource_grade_header(spec)
        row[header] = weighted_mean(subset[spec.column], subset[config.mass_column]) if not subset.empty and spec.column in subset.columns else float("nan")

    for metal_label in ["Au", "Ag"]:
        spec = _find_metal_grade_spec(specs, metal_label)
        if spec and spec.column in subset.columns and not subset.empty:
            metal = contained_metal(subset[spec.column], subset[config.mass_column], _effective_grade_unit(spec))
            row[f"{metal_label} (oz)"] = metal[0] if metal and metal[1] == "oz" else float("nan")
        else:
            row[f"{metal_label} (oz)"] = float("nan")

    cu_spec = _find_metal_grade_spec(specs, "Cu")
    if cu_spec and cu_spec.column in subset.columns and not subset.empty:
        cu_tonnes = contained_metal(subset[cu_spec.column], subset[config.mass_column], _effective_grade_unit(cu_spec))
        row["Cu (lb)"] = cu_tonnes[0] * 2204.62262185 if cu_tonnes and cu_tonnes[1] == "t" else float("nan")
    else:
        row["Cu (lb)"] = float("nan")
    return row

def _resource_tabulation(data: pd.DataFrame, config: ModelConfig) -> pd.DataFrame:
    specs = _preferred_grade_specs(config)
    category_col = config.column_for_role("Category")
    if not category_col or category_col not in data.columns:
        return pd.DataFrame([_resource_row("Grand Total", data, config, specs)])

    measured = _category_mask(data, category_col, "measured")
    indicated = _category_mask(data, category_col, "indicated")
    inferred = _category_mask(data, category_col, "inferred")
    inventory = _category_mask(data, category_col, "inventory")
    unclassified = _category_mask(data, category_col, "unclassified")
    grade_control = _category_mask(data, category_col, "grade_control")

    mi_mask = measured | indicated
    total_mask = measured | indicated | inferred
    rows = []

    if _is_grade_control_category_column(category_col) and grade_control.any():
        rows.append(_resource_row("Grade Control", data[grade_control], config, specs))
        mi_mask = grade_control | measured | indicated
        total_mask = grade_control | measured | indicated | inferred

    rows.extend([
        _resource_row("Measured", data[measured], config, specs),
        _resource_row("Indicated", data[indicated], config, specs),
        _resource_row("M&I", data[mi_mask], config, specs),
        _resource_row("Inferred", data[inferred], config, specs),
        _resource_row("Total", data[total_mask], config, specs),
        _resource_row("Inventory", data[inventory], config, specs),
        _resource_row("Unclassified", data[unclassified], config, specs),
        _resource_row("Grand Total", data, config, specs),
    ])
    return pd.DataFrame(rows)


def _barrick_number_formatters(table: pd.DataFrame, config: ModelConfig) -> dict[str, str]:
    formatters: dict[str, str] = {}
    for column in table.columns:
        lower = column.casefold()
        if _is_year_like_column(column):
            formatters[column] = "{:.0f}"
        elif column.startswith("Tonnes") or column.startswith("Tonnage"):
            formatters[column] = f"{{:,.{config.tonnage_decimals}f}}"
        elif "volume" in lower and pd.api.types.is_numeric_dtype(table[column]):
            formatters[column] = "{:,.0f}"
        elif column.endswith("(oz)") or column.endswith("(lb)") or "contained" in lower:
            formatters[column] = "{:,.0f}"
        elif column != "Category" and pd.api.types.is_numeric_dtype(table[column]):
            formatters[column] = f"{{:,.{config.grade_decimals}f}}"
    return formatters


def _is_year_like_column(column: Any) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(column).casefold()).strip("_")
    return normalized in {"year", "years", "anio", "ano"} or normalized.endswith("_year")


def _table_number_formatters(table: pd.DataFrame, precision: int = 3) -> dict[str, str]:
    formatters: dict[str, str] = {}
    for column in table.columns:
        lower = str(column).casefold()
        if _is_year_like_column(column) and pd.api.types.is_numeric_dtype(table[column]):
            formatters[column] = "{:.0f}"
        elif "volume" in lower and pd.api.types.is_numeric_dtype(table[column]):
            formatters[column] = "{:,.0f}"
        elif pd.api.types.is_numeric_dtype(table[column]):
            formatters[column] = f"{{:,.{precision}f}}"
    return formatters


def _left_aligned_table_style(
    table: pd.DataFrame,
    *,
    formatters: dict[str, str] | None = None,
    na_rep: str = "-",
) -> pd.io.formats.style.Styler:
    """Left-align headers and cell values for selected descriptive tables."""
    styler = table.style
    if formatters:
        styler = styler.format(formatters, na_rep=na_rep)
    return styler.set_properties(**{"text-align": "left"}).set_table_styles(
        [
            {"selector": "th", "props": [("text-align", "left")]},
            {"selector": "td", "props": [("text-align", "left")]},
        ]
    )


def _text_table_column_config(
    table: pd.DataFrame,
    *,
    narrow_columns: tuple[str, ...] = (),
    medium_columns: tuple[str, ...] = (),
    wide_columns: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Return Streamlit column widths that reserve space for narrative text."""
    config: dict[str, Any] = {}
    narrow = {str(column) for column in narrow_columns}
    medium = {str(column) for column in medium_columns}
    wide = {str(column) for column in wide_columns}

    narrative_tokens = (
        "detail",
        "comment",
        "note",
        "description",
        "message",
        "rule",
        "values / range",
    )

    for column in table.columns:
        name = str(column)
        normalized = name.casefold()
        if name in wide or any(token in normalized for token in narrative_tokens):
            config[column] = st.column_config.TextColumn(width="large")
        elif name in narrow:
            config[column] = st.column_config.TextColumn(width="small")
        elif name in medium:
            config[column] = st.column_config.TextColumn(width="medium")

    return config


def _render_barrick_table(table: pd.DataFrame, config: ModelConfig) -> None:
    if table.empty:
        st.info("No data available for the current filters.")
        return

    def row_style(row: pd.Series) -> list[str]:
        category = str(row.get("Category", ""))
        if any(token in category for token in ["M&I", "Total", "Grand Total"]):
            return ["color:#0000A8;font-weight:700;" for _ in row]
        return ["" for _ in row]

    styler = (
        table.style.format(_barrick_number_formatters(table, config), na_rep="-")
        .hide(axis="index")
        .apply(row_style, axis=1)
        .set_table_attributes('class="bm-resource-table"')
        .set_table_styles(
            [
                {"selector": "th", "props": [("background-color", "#005A87"), ("color", "white"), ("font-weight", "800"), ("text-align", "center")]},
                {"selector": "td", "props": [("border", "1px solid #D9E2EC"), ("padding", "4px 7px"), ("text-align", "right")]},
                {"selector": "td:first-child", "props": [("text-align", "left")]},
                {"selector": "table", "props": [("border-collapse", "collapse"), ("width", "auto"), ("font-size", "0.86rem")]},
            ]
        )
    )
    st.markdown(f'<div class="bm-table-wrap">{styler.to_html()}</div>', unsafe_allow_html=True)


def _metric_tonnage(data: pd.DataFrame, config: ModelConfig) -> float:
    return total_tonnage(data, config) / tonnage_divisor(config.tonnage_unit)


def _resource_calculation_audit(source_data: pd.DataFrame, scoped_data: pd.DataFrame, final_data: pd.DataFrame, config: ModelConfig, destination_mode: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    ton_col = f"Tonnes ({config.tonnage_unit})"

    def add_row(step: str, frame: pd.DataFrame, note: str) -> None:
        rows.append(
            {
                "Step": step,
                "Rows": int(len(frame)),
                ton_col: _metric_tonnage(frame, config) if not frame.empty else 0.0,
                "Note": note,
            }
        )

    add_row("01. Uploaded/cleaned model", source_data, "Before transverse BLK_MODEL and destination exclusions.")
    add_row("02. Master scope", scoped_data, "BLK_MODEL, Year scope and invalid/CUT/NONE destinations are applied.")
    add_row("03. Resource tabulation", final_data, f"Destination mode = {destination_mode}.")
    excluded_by_destination_mode = scoped_data.loc[scoped_data.index.difference(final_data.index)]
    if not excluded_by_destination_mode.empty:
        add_row(
            "Excluded by resource destination mode",
            excluded_by_destination_mode,
            f"Rows removed by the selected resource-table destination mode ({destination_mode}).",
        )

    dest_col = config.column_for_role("Destination")
    if dest_col and dest_col in source_data.columns:
        invalid = source_data[~_valid_destination_mask(source_data, config)]
        add_row("Excluded invalid destinations", invalid, "Rows with NONE, CUT suffixes or destinations outside H1/H2/L1/L2/L3/M1/M2/M3/MW1/MW2/W1/W2.")

    blk_col = _block_model_column(config, source_data)
    if blk_col and blk_col in source_data.columns:
        blk_numeric = pd.to_numeric(source_data[blk_col], errors="coerce")
        ignored = source_data[blk_numeric.eq(0)]
        add_row("Excluded BLK_MODEL = 0", ignored, "Rows explicitly flagged as not part of the model.")


    return pd.DataFrame(rows)


ORE_DESTINATION_CODES = ["h1", "h2", "l1", "l2", "l3", "m1", "m2", "m3"]
WASTE_DESTINATION_CODES = ["w1", "w2", "mw1", "mw2"]
PRIMARY_ORE_DESTINATION_CODES = ["h1", "h2", "l1", "l2", "l3"]
OPTIONAL_ORE_DESTINATION_CODES = ["m1", "m2", "m3"]


def _data_by_destination_codes(data: pd.DataFrame, config: ModelConfig, codes: list[str]) -> pd.DataFrame:
    dest_col = config.column_for_role("Destination")
    if not dest_col or dest_col not in data.columns or data.empty:
        return data.iloc[0:0].copy()
    normalized = data[dest_col].map(_normalize_destination_code)
    return data[normalized.isin(codes)].copy()


def _tonnes_for_frame(data: pd.DataFrame, config: ModelConfig) -> float:
    if data.empty or config.mass_column not in data.columns:
        return 0.0
    return float(pd.to_numeric(data[config.mass_column], errors="coerce").fillna(0).clip(lower=0).sum())


def _display_tonnage_value(tonnes: float, config: ModelConfig) -> float:
    return tonnes / tonnage_divisor(config.tonnage_unit)


def _format_destination_value(value: float | int | None, decimals: int = 3) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):,.{decimals}f}"


def _format_grade_value(value: float | int | None, decimals: int) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):,.{decimals}f}"


def _spec_by_canonical_label(config: ModelConfig, label: str) -> GradeSpec | None:
    specs = _preferred_grade_specs(config)
    return next((spec for spec in specs if _canonical_grade_label(spec).casefold() == label.casefold()), None)


def _weighted_grade_for_label(data: pd.DataFrame, config: ModelConfig, label: str) -> tuple[str, float | None]:
    spec = _spec_by_canonical_label(config, label)
    if not spec or spec.column not in data.columns or data.empty:
        return "", None
    return _effective_grade_unit(spec), weighted_mean(data[spec.column], data[config.mass_column])


def _contained_ounces_for_label(data: pd.DataFrame, config: ModelConfig, label: str) -> float | None:
    spec = _spec_by_canonical_label(config, label)
    if not spec or spec.column not in data.columns or data.empty:
        return None
    metal = contained_metal(data[spec.column], data[config.mass_column], _effective_grade_unit(spec))
    if not metal:
        return None
    value, unit = metal
    return float(value) if unit == "oz" else None


def _destination_summary_table(data: pd.DataFrame, config: ModelConfig, model_name: str) -> pd.DataFrame:
    dest_col = config.column_for_role("Destination")
    value_col = model_name or "Model Name"
    if not dest_col or dest_col not in data.columns:
        return pd.DataFrame(columns=["Model", "Unit", value_col, "__row_type__"])

    work = data.copy()
    work["__dest_code__"] = work[dest_col].map(_normalize_destination_code)

    ore_codes = list(ORE_DESTINATION_CODES)
    waste_codes = list(WASTE_DESTINATION_CODES)
    ore_data = work[work["__dest_code__"].isin(ore_codes)].copy()
    waste_data = work[work["__dest_code__"].isin(waste_codes)].copy()

    ore_tonnes = _tonnes_for_frame(ore_data, config)
    waste_tonnes = _tonnes_for_frame(waste_data, config)
    total_tonnes = ore_tonnes + waste_tonnes
    strip_ratio = waste_tonnes / ore_tonnes if ore_tonnes > 0 else float("nan")
    tonnage_decimals = max(2, int(config.tonnage_decimals))

    rows: list[dict[str, Any]] = []

    def add(label: str, unit: str, value: str, row_type: str = "normal") -> None:
        rows.append({"Model": label, "Unit": unit, value_col: value, "__row_type__": row_type})

    def tonnes_by_code(code: str) -> float:
        subset = work[work["__dest_code__"].eq(code)]
        return _display_tonnage_value(_tonnes_for_frame(subset, config), config)

    add("Ore tonnes", config.tonnage_unit, _format_destination_value(_display_tonnage_value(ore_tonnes, config), tonnage_decimals), "section")
    for code in PRIMARY_ORE_DESTINATION_CODES:
        add(code, config.tonnage_unit, _format_destination_value(tonnes_by_code(code), tonnage_decimals))
    for code in OPTIONAL_ORE_DESTINATION_CODES:
        if work["__dest_code__"].eq(code).any():
            add(code, config.tonnage_unit, _format_destination_value(tonnes_by_code(code), tonnage_decimals))

    add("Waste tonnes", config.tonnage_unit, _format_destination_value(_display_tonnage_value(waste_tonnes, config), tonnage_decimals), "section")
    for code in WASTE_DESTINATION_CODES:
        add(code, config.tonnage_unit, _format_destination_value(tonnes_by_code(code), tonnage_decimals))

    add("Total tonnes", config.tonnage_unit, _format_destination_value(_display_tonnage_value(total_tonnes, config), tonnage_decimals), "section")
    add("Strip Ratio", "t/t", _format_destination_value(strip_ratio, 2), "separator")

    grade_specs = [
        ("Au Grade", "Au"),
        ("Ag Grade", "Ag"),
        ("Cu Grade", "Cu"),
        ("Stot", "Stot"),
        ("S2 Grade", "S2"),
        ("Ctot", "Ctot"),
        ("OC Grade", "OC"),
    ]
    for row_label, canonical in grade_specs:
        unit, value = _weighted_grade_for_label(ore_data, config, canonical)
        add(row_label, unit or "-", _format_grade_value(value, int(config.grade_decimals)))

    for metal_label in ["Au", "Ag"]:
        total_oz = _contained_ounces_for_label(ore_data, config, metal_label)
        add(f"{metal_label} Metal", "Moz", _format_destination_value((total_oz or 0.0) / 1_000_000, 3), "section")
        for code in PRIMARY_ORE_DESTINATION_CODES:
            subset = work[work["__dest_code__"].eq(code)]
            ounces = _contained_ounces_for_label(subset, config, metal_label)
            add(code, "Moz", _format_destination_value((ounces or 0.0) / 1_000_000, 3))
        for code in OPTIONAL_ORE_DESTINATION_CODES:
            if work["__dest_code__"].eq(code).any():
                subset = work[work["__dest_code__"].eq(code)]
                ounces = _contained_ounces_for_label(subset, config, metal_label)
                add(code, "Moz", _format_destination_value((ounces or 0.0) / 1_000_000, 3))

    return pd.DataFrame(rows)


def _render_destination_summary_table(table: pd.DataFrame) -> None:
    if table.empty:
        st.info("No destination summary can be calculated for the current filters.")
        return

    row_types = table["__row_type__"].tolist() if "__row_type__" in table.columns else ["normal"] * len(table)
    visible = table.drop(columns=["__row_type__"], errors="ignore")

    def row_style(row: pd.Series) -> list[str]:
        row_type = row_types[row.name] if row.name < len(row_types) else "normal"
        if row_type == "section":
            return ["background-color:#E6E6E6;color:#111111;font-weight:800;" for _ in row]
        if row_type == "separator":
            return ["border-top:1px solid #D9E2EC;" for _ in row]
        return ["" for _ in row]

    styler = (
        visible.style
        .hide(axis="index")
        .apply(row_style, axis=1)
        .set_table_attributes('class="bm-resource-table"')
        .set_table_styles(
            [
                {"selector": "th", "props": [("background-color", "#005A87"), ("color", "white"), ("font-weight", "800"), ("text-align", "center")]},
                {"selector": "td", "props": [("border", "1px solid #D9E2EC"), ("padding", "4px 7px"), ("text-align", "right")]},
                {"selector": "td:first-child", "props": [("text-align", "left")]},
                {"selector": "td:nth-child(2)", "props": [("text-align", "center")]},
                {"selector": "table", "props": [("border-collapse", "collapse"), ("width", "auto"), ("font-size", "0.86rem")]},
            ]
        )
    )
    st.markdown(f'<div class="bm-table-wrap">{styler.to_html()}</div>', unsafe_allow_html=True)


def _plot_ore_kt_by_phase(data: pd.DataFrame, config: ModelConfig, model_name: str) -> None:
    phase_col = _phase_column(config, data)
    dest_col = config.column_for_role("Destination")
    if not phase_col or phase_col not in data.columns:
        st.info("Configure a Phase variable to activate the Ore Kt by Phase chart.")
        return
    if not dest_col or dest_col not in data.columns:
        st.info("Configure a Destination variable to activate the Ore Kt by Phase chart.")
        return

    work = data.copy()
    work["__dest_code__"] = work[dest_col].map(_normalize_destination_code)
    ore = work[work["__dest_code__"].isin(ORE_DESTINATION_CODES)].copy()
    if ore.empty:
        st.info("No ore tonnes are available for the current filters.")
        return

    ore[config.mass_column] = pd.to_numeric(ore[config.mass_column], errors="coerce").fillna(0).clip(lower=0)
    table = ore.groupby(phase_col, dropna=False, observed=True)[config.mass_column].sum().reset_index()
    table["Ore Kt"] = table[config.mass_column] / 1_000.0
    table = table.drop(columns=[config.mass_column])
    table[phase_col] = table[phase_col].astype(str)

    fig = px.bar(
        table,
        x=phase_col,
        y="Ore Kt",
        title="Ore Kt",
        labels={phase_col: "Phase", "Ore Kt": "Ore Kt"},
    )
    fig.update_traces(marker_color="#03547C", marker_line_color="white", marker_line_width=0.5, name=model_name)
    fig.update_layout(
        height=470,
        showlegend=False,
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        margin={"l": 45, "r": 25, "t": 65, "b": 85},
    )
    fig.update_xaxes(tickangle=-45)
    fig.update_yaxes(showgrid=True, gridcolor="#D9D9D9", zeroline=True, zerolinecolor="#D9D9D9")
    st.plotly_chart(fig, use_container_width=True)


def _comparison_ore_kt_by_phase_table(bundles: dict[str, ModelBundle], selected_model_names: list[str]) -> pd.DataFrame:
    """Return ore tonnes by unified Phase/Pit_Phase and model, scaled to Kt."""
    rows: list[dict[str, Any]] = []

    for model_name in selected_model_names:
        bundle = bundles.get(model_name)
        if bundle is None:
            continue
        config = bundle.config
        data = bundle.data
        phase_col = _phase_column(config, data)
        dest_col = config.column_for_role("Destination")

        if not phase_col or phase_col not in data.columns or not dest_col or dest_col not in data.columns:
            continue
        if config.mass_column not in data.columns or data.empty:
            continue

        work = data.copy()
        work["__dest_code__"] = work[dest_col].map(_normalize_destination_code)
        ore = work[work["__dest_code__"].isin(ORE_DESTINATION_CODES)].copy()
        if ore.empty:
            continue

        ore[config.mass_column] = pd.to_numeric(ore[config.mass_column], errors="coerce").fillna(0).clip(lower=0)
        grouped = ore.groupby(phase_col, dropna=False, observed=True)[config.mass_column].sum().reset_index()
        grouped["Phase"] = grouped[phase_col].astype(str).str.strip()
        grouped["Model"] = model_name
        grouped["Ore Kt"] = grouped[config.mass_column] / 1_000.0
        rows.extend(grouped[["Model", "Phase", "Ore Kt"]].to_dict("records"))

    if not rows:
        return pd.DataFrame(columns=["Model", "Phase", "Ore Kt"])

    table = pd.DataFrame(rows)
    phase_order = sorted(table["Phase"].dropna().unique().tolist(), key=_natural_phase_sort_key)
    table["Phase"] = pd.Categorical(table["Phase"], categories=phase_order, ordered=True)
    table["Model"] = pd.Categorical(table["Model"], categories=selected_model_names, ordered=True)
    return table.sort_values(["Phase", "Model"]).reset_index(drop=True)


def _plot_comparison_ore_kt_by_phase(bundles: dict[str, ModelBundle], selected_model_names: list[str]) -> None:
    table = _comparison_ore_kt_by_phase_table(bundles, selected_model_names)
    if table.empty:
        st.info("Configure Phase/Pit_Phase, Destination and tonnage columns in the selected models to activate the Ore Kt by Phase comparison chart.")
        return

    model_color_sequence = ["#03547C", "#A39161", *MODEL_COLORS]
    color_map = {
        model_name: model_color_sequence[index % len(model_color_sequence)]
        for index, model_name in enumerate(selected_model_names)
    }

    fig = px.bar(
        table,
        x="Phase",
        y="Ore Kt",
        color="Model",
        barmode="group",
        title="Ore Kt by Phase",
        labels={"Phase": "Phase", "Ore Kt": "Ore Kt"},
        color_discrete_map=color_map,
        category_orders={"Model": selected_model_names, "Phase": list(table["Phase"].cat.categories)},
    )
    fig.update_traces(marker_line_color="white", marker_line_width=0.5)
    fig.update_layout(
        height=500,
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        legend_title_text="Model",
        margin={"l": 55, "r": 25, "t": 65, "b": 95},
    )
    fig.update_xaxes(tickangle=-45, tickfont={"size": 10})
    fig.update_yaxes(showgrid=True, gridcolor="#D9D9D9", zeroline=True, zerolinecolor="#D9D9D9")
    st.plotly_chart(fig, use_container_width=True)


def _render_resource_by_destination(
    bundle: ModelBundle,
    model_name: str,
    base_data: pd.DataFrame | None = None,
    base_filters: dict[str, Any] | None = None,
) -> None:
    config = bundle.config
    st.subheader("Tabulation by Destination")

    source_data = base_data.copy() if base_data is not None else bundle.data.copy()
    filtered = _apply_global_scope(source_data, config)
    filters = dict(base_filters or {})
    _scope_caption(source_data, filtered, config)

    if filtered.empty:
        st.warning("No rows match the current filters.")
        return

    table = _destination_summary_table(filtered, config, model_name)
    _render_destination_summary_table(table)
    _render_table_filter_note(
        _table_filter_note_items(
            config,
            source_data=source_data,
            final_data=filtered,
            filters=filters,
            model_name=model_name,
            extra_items=["Destination table grouping: ore, waste, grades and contained metal by destination code"],
        )
    )
    _add_scene_button(f"{model_name} - resource tabulation by destination", "Resource by destination", [model_name], table.drop(columns=["__row_type__"], errors="ignore"), filters)

    st.markdown("#### Ore Kt by Phase")
    _plot_ore_kt_by_phase(filtered, config, model_name)


def _destination_table_with_row_keys(table: pd.DataFrame) -> pd.DataFrame:
    """Add stable row keys so repeated labels such as h1 under tonnes and metal sections align correctly."""
    if table.empty:
        return table.copy()

    keyed = table.copy().reset_index(drop=True)
    current_section = ""
    row_keys: list[str] = []
    for _, row in keyed.iterrows():
        label = str(row.get("Model", "")).strip()
        unit = str(row.get("Unit", "")).strip()
        row_type = str(row.get("__row_type__", "normal"))
        if row_type == "section":
            current_section = label
            row_key = f"section::{label}::{unit}"
        else:
            row_key = f"{current_section}::{label}::{unit}"
        row_keys.append(row_key)

    keyed["__row_key__"] = row_keys
    return keyed


def _numeric_from_destination_value(value: Any) -> float | None:
    """Convert table display values back to numbers for relative-difference calculations."""
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text in {"-", "N/A", "nan", "NaN"}:
        return None
    text = text.replace(",", "").replace("%", "")
    try:
        return float(text)
    except ValueError:
        return None


def _format_relative_difference_pct(value: Any, reference_value: Any) -> str:
    value_number = _numeric_from_destination_value(value)
    reference_number = _numeric_from_destination_value(reference_value)
    if value_number is None or reference_number is None:
        return "-"
    if abs(reference_number) < 1e-12:
        return "0%" if abs(value_number) < 1e-12 else "-"
    return f"{((value_number - reference_number) / reference_number) * 100:,.0f}%"


def _add_relative_difference_columns(rows: list[dict[str, Any]], model_names: list[str], reference_model_name: str | None) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()

    table = pd.DataFrame(rows)
    if not reference_model_name or reference_model_name not in model_names or reference_model_name not in table.columns:
        return table

    ordered_columns = ["Model", "Unit"]
    for model_name in model_names:
        ordered_columns.append(model_name)
        if model_name == reference_model_name:
            continue
        diff_col = f"{model_name} Δ%"
        table[diff_col] = table.apply(
            lambda row, model=model_name: _format_relative_difference_pct(row.get(model), row.get(reference_model_name)),
            axis=1,
        )
        ordered_columns.append(diff_col)

    if "__row_type__" in table.columns:
        ordered_columns.append("__row_type__")

    return table[[column for column in ordered_columns if column in table.columns]]


def _destination_comparison_table(bundles: dict[str, ModelBundle], reference_model_name: str | None = None) -> pd.DataFrame:
    """Build the vertical destination table with one value column per selected model plus Δ% columns vs reference."""
    row_meta: dict[str, dict[str, Any]] = {}
    model_values: dict[str, dict[str, Any]] = {}

    for model_name, bundle in bundles.items():
        table = _destination_summary_table(bundle.data, bundle.config, model_name)
        if table.empty:
            model_values[model_name] = {}
            continue

        keyed = _destination_table_with_row_keys(table)
        value_columns = [
            column for column in keyed.columns
            if column not in {"Model", "Unit", "__row_type__", "__row_key__"}
        ]
        if not value_columns:
            model_values[model_name] = {}
            continue
        value_col = value_columns[0]

        values: dict[str, Any] = {}
        for _, row in keyed.iterrows():
            row_key = str(row["__row_key__"])
            if row_key not in row_meta:
                row_meta[row_key] = {
                    "Model": row.get("Model", ""),
                    "Unit": row.get("Unit", ""),
                    "__row_type__": row.get("__row_type__", "normal"),
                }
            values[row_key] = row.get(value_col, "-")
        model_values[model_name] = values

    model_names = list(bundles.keys())
    rows: list[dict[str, Any]] = []
    for row_key, meta in row_meta.items():
        row = dict(meta)
        for model_name in model_names:
            row[model_name] = model_values.get(model_name, {}).get(row_key, "-")
        rows.append(row)

    category_rows = [
        ("Grade Control tonnes", "grade_control"),
        ("Measured tonnes", "measured"),
        ("Indicated tonnes", "indicated"),
        ("Inferred tonnes", "inferred"),
    ]
    default_unit = next(iter(bundles.values())).config.tonnage_unit if bundles else "Mt"

    for row_index, (row_label, category_key) in enumerate(category_rows):
        row = {
            "Model": row_label,
            "Unit": default_unit,
            "__row_type__": "separator" if row_index == 0 else "normal",
        }
        for model_name, bundle in bundles.items():
            category_col = bundle.config.column_for_role("Category")
            if not category_col or category_col not in bundle.data.columns:
                row[model_name] = "-"
                continue
            mask = _category_mask(bundle.data, category_col, category_key)
            tonnes = _tonnes_for_frame(bundle.data[mask], bundle.config)
            display_tonnes = _display_tonnage_value(tonnes, bundle.config)
            decimals = max(2, int(bundle.config.tonnage_decimals))
            row[model_name] = _format_destination_value(display_tonnes, decimals)
        rows.append(row)

    return _add_relative_difference_columns(rows, model_names, reference_model_name)


def _render_comparison_resource_by_destination(bundles: dict[str, ModelBundle], selected_model_names: list[str]) -> None:
    st.subheader("Tabulation by Destination")
    st.caption("Select the reference model used to calculate the relative difference columns. Δ% = (model - reference) / reference × 100.")

    reference_model = st.selectbox(
        "Reference model",
        selected_model_names,
        index=0,
        key="comparison_destination_reference_model",
        help="Relative differences are calculated against this model and reported without decimals.",
    )
    st.info("Note: comparing more than 5 models can affect the correct table presentation because of horizontal space limits.")

    table = _destination_comparison_table(bundles, reference_model)
    if table.empty:
        st.info("No destination comparison table can be calculated for the selected models and filters.")
        return

    _render_destination_summary_table(table)
    _render_table_filter_note(
        _comparison_filter_note_items(
            bundles,
            selected_model_names,
            reference_model,
            extra_items=["Comparison table grouping: ore, waste, grades, contained metal and resource-category tonnes by model"],
        )
    )
    _add_scene_button(
        "Comparison - resource tabulation by destination",
        "Comparison resource by destination",
        selected_model_names,
        table.drop(columns=["__row_type__"], errors="ignore"),
        {"Scope": "Comparison master scope", "Reference model": reference_model},
    )

    st.markdown("#### Ore Kt by Phase")
    _plot_comparison_ore_kt_by_phase(bundles, selected_model_names)


def _dashboard_filters(bundle: ModelBundle, key_prefix: str) -> tuple[pd.DataFrame, dict[str, list[Any]]]:
    """Backward-compatible filter helper kept for older pages."""
    config = bundle.config
    data = bundle.data.copy()
    filters: dict[str, list[Any]] = {}
    for role in ["Year", "Pit"]:
        column = config.column_for_role(role)
        if column and column in data.columns:
            values = sorted(data[column].dropna().unique().tolist())
            filters[column] = values
    return apply_categorical_filters(data, filters), filters


def _sidebar_evaluation_filters(bundle: ModelBundle, key_prefix: str, base_data: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    config = bundle.config
    data = base_data.copy() if base_data is not None else bundle.data.copy()
    filters: dict[str, Any] = {}

    st.sidebar.markdown(
        """
        <div class="bm-side-banner">
            <div class="bm-banner-kicker">Filters</div>
            <div class="bm-banner-title">Evaluation controls</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    filtered = data
    for role in ["Pit", "Mettype", "Category"]:
        column = config.column_for_role(role)
        if not column or column not in data.columns:
            continue
        values = sorted(data[column].dropna().unique().tolist(), key=lambda item: str(item))
        if not values:
            continue
        selected = st.sidebar.multiselect(
            config.category_label(column),
            values,
            default=values,
            key=f"{key_prefix}_sidebar_{role}_{_safe_key(column)}",
        )
        filters[column] = selected
        if selected:
            filtered = filtered[filtered[column].isin(selected)]

    bench_col = config.column_for_role("Bench")
    if bench_col and bench_col in data.columns:
        bench_numeric = pd.to_numeric(data[bench_col], errors="coerce")
        if bench_numeric.notna().any():
            min_bench = int(bench_numeric.min())
            max_bench = int(bench_numeric.max())
            selected_range = st.sidebar.slider(
                config.category_label(bench_col),
                min_value=min_bench,
                max_value=max_bench,
                value=(min_bench, max_bench),
                step=10 if max_bench - min_bench >= 10 else 1,
                key=f"{key_prefix}_sidebar_bench_range",
            )
            filters[bench_col] = list(selected_range)
            filtered_numeric = pd.to_numeric(filtered[bench_col], errors="coerce")
            filtered = filtered[(filtered_numeric >= selected_range[0]) & (filtered_numeric <= selected_range[1])]

    st.sidebar.caption(f"Rows after filters: **{len(filtered):,}**")
    return filtered, filters


def _apply_destination_mode(data: pd.DataFrame, config: ModelConfig, mode: str) -> pd.DataFrame:
    dest_col = config.column_for_role("Destination")
    if not dest_col or dest_col not in data.columns or data.empty:
        return data

    dest_code = data[dest_col].map(_normalize_destination_code)
    return data[dest_code.isin(DESTINATION_MODE_CODES.get(mode, VALID_DESTINATIONS))]

def _bench_order(values: pd.Series) -> list[Any]:
    clean = [value for value in values.dropna().unique().tolist()]
    try:
        return sorted(clean, key=lambda x: float(x), reverse=True)
    except (TypeError, ValueError):
        return sorted(clean, key=lambda x: str(x), reverse=True)


def _group_tonnes(data: pd.DataFrame, config: ModelConfig, group_cols: list[str]) -> pd.DataFrame:
    cols = [col for col in group_cols if col and col in data.columns]
    if not cols or data.empty:
        return pd.DataFrame()
    work = data.copy()
    work[config.mass_column] = work[config.mass_column].clip(lower=0)
    table = work.groupby(cols, dropna=False, observed=True)[config.mass_column].sum().reset_index()
    table[tonnage_column_name(config)] = table[config.mass_column] / tonnage_divisor(config.tonnage_unit)
    return table.drop(columns=[config.mass_column])


def _plot_mettype_distribution(data: pd.DataFrame, config: ModelConfig) -> None:
    _plot_role_distribution(data, config, "Mettype", "Met-Type Ore Tonnes Distribution", METTYPE_COLORS)


def _plot_role_distribution(data: pd.DataFrame, config: ModelConfig, role: str, title: str, color_map: dict[str, str]) -> None:
    role_col = config.column_for_role(role)
    if not role_col or role_col not in data.columns or data.empty:
        st.info(f"Configure a {role} variable to activate this chart.")
        return
    table = _group_tonnes(data, config, [role_col])
    if table.empty:
        return
    ton_col = tonnage_column_name(config)
    label = config.category_label(role_col)
    table[label] = table[role_col].astype(str).str.strip()
    if role == "Mettype":
        table[label] = table[label].str.upper()
    if role == "Destination":
        table[label] = table[label].map(_display_destination)
    if role == "Category":
        table[label] = table[label].map(_display_resource_category)
        order_map = {name: index for index, name in enumerate(RESOURCE_CATEGORY_ORDER)}
        table["__category_order__"] = table[label].map(order_map).fillna(len(order_map))
        table = table.sort_values("__category_order__").drop(columns="__category_order__")
    fig = px.pie(
        table,
        names=label,
        values=ton_col,
        hole=0.58,
        title=title,
        color=label,
        color_discrete_map=color_map,
    )
    fig.update_traces(textinfo="percent+label")
    fig.update_layout(height=370, margin={"l": 20, "r": 20, "t": 60, "b": 20})
    st.plotly_chart(fig, use_container_width=True)

def _plot_stacked_by_bench(data: pd.DataFrame, config: ModelConfig, color_role: str, title: str, color_map: dict[str, str], percent: bool = True) -> None:
    bench_col = config.column_for_role("Bench")
    color_col = config.column_for_role(color_role)
    if not bench_col or not color_col or bench_col not in data.columns or color_col not in data.columns or data.empty:
        st.info(f"Configure Bench and {color_role} roles to activate this chart.")
        return
    table = _group_tonnes(data, config, [bench_col, color_col])
    if table.empty:
        return
    ton_col = tonnage_column_name(config)
    bench_label = config.category_label(bench_col)
    color_label = config.category_label(color_col)
    table = table.rename(columns={bench_col: bench_label, color_col: color_label})
    if color_role == "Category":
        table[color_label] = table[color_label].map(_display_resource_category)
        color_order = {color_label: RESOURCE_CATEGORY_ORDER}
    elif color_role == "Destination":
        table[color_label] = table[color_label].map(_display_destination)
        color_order = None
    else:
        table[color_label] = table[color_label].astype(str).str.strip().str.upper()
        color_order = None

    numeric_bench = pd.to_numeric(table[bench_label], errors="coerce")
    use_numeric_axis = numeric_bench.notna().all()
    if use_numeric_axis:
        table["__bench_x__"] = numeric_bench.astype(float)
        x_col = "__bench_x__"
        category_orders = None
    else:
        x_col = bench_label
        category_orders = {bench_label: _bench_order(table[bench_label])}
    if color_order:
        category_orders = {**(category_orders or {}), **color_order}

    fig = px.bar(
        table,
        x=x_col,
        y=ton_col,
        color=color_label,
        title=title,
        color_discrete_map=color_map,
        category_orders=category_orders,
    )
    if percent:
        fig.update_layout(barnorm="percent", yaxis_title="Distribution (%)")
    else:
        fig.update_layout(yaxis_title=ton_col)
    fig.update_layout(
        height=500,
        barmode="stack",
        legend_title_text=color_label,
        margin={"l": 25, "r": 20, "t": 65, "b": 45},
    )
    if use_numeric_axis:
        fig.update_xaxes(title_text="BENCH", dtick=10, autorange="reversed", tickformat=".0f")
    else:
        fig.update_xaxes(title_text="BENCH")
    fig.update_traces(marker_line_width=0.5, marker_line_color="white")
    st.plotly_chart(fig, use_container_width=True)



def _active_year_scope_short_label() -> str:
    mode = str(st.session_state.get("master_year_scope", "LOM"))
    selected = st.session_state.get("master_selected_years", [])
    if mode == "Single year" and selected:
        return str(selected[0])
    if mode == "Custom years":
        return "Custom years"
    if mode == "All years":
        return "All years"
    return mode or "LOM"


def _metal_at_risk_display_unit(max_ounces: float) -> tuple[str, float, str]:
    """Return a dynamic Au-ounce display unit for the selected period and filters."""
    if max_ounces >= 1_000_000:
        return "Moz", 1_000_000.0, ",.2f"
    if max_ounces >= 1_000:
        return "Koz", 1_000.0, ",.0f"
    return "oz", 1.0, ",.0f"


def _plot_metal_at_risk_by_bench(data: pd.DataFrame, config: ModelConfig) -> None:
    bench_col = config.column_for_role("Bench")
    category_col = config.column_for_role("Category")
    au_spec = _spec_by_canonical_label(config, "Au")

    if not bench_col or bench_col not in data.columns:
        st.info("Configure a Bench variable to activate the Metal at Risk chart.")
        return
    if not category_col or category_col not in data.columns:
        st.info("Configure a Category variable to activate the Metal at Risk chart.")
        return
    if not au_spec or au_spec.column not in data.columns:
        st.info("Configure an Au grade variable to activate the Metal at Risk chart.")
        return
    if data.empty:
        st.info("No data are available for the current filters.")
        return

    category_specs = [
        ("grade_control", "Grade Control"),
        ("measured", "Measured"),
        ("indicated", "Indicated (At Risk)"),
    ]
    au_unit = _effective_grade_unit(au_spec)
    rows: list[dict[str, Any]] = []

    for category_key, display_label in category_specs:
        subset = data[_category_mask(data, category_col, category_key)].copy()
        if subset.empty:
            continue
        for bench, group in subset.groupby(bench_col, dropna=False, observed=True):
            metal = contained_metal(group[au_spec.column], group[config.mass_column], au_unit)
            ounces = float(metal[0]) if metal and metal[1] == "oz" else 0.0
            rows.append({"Bench": bench, "Category": display_label, "Au oz": ounces})

    if not rows:
        st.info("No Grade Control, Measured or Indicated Au ounces are available for the current filters.")
        return

    table = pd.DataFrame(rows)
    table["__bench_numeric__"] = pd.to_numeric(table["Bench"], errors="coerce")
    use_numeric_bench_axis = bool(table["__bench_numeric__"].notna().all())
    bench_order = _bench_order(table["Bench"])
    bench_labels = [str(value) for value in bench_order]
    if not use_numeric_bench_axis:
        table["Bench"] = table["Bench"].astype(str)

    # Scale the chart dynamically from the selected period/filter bench totals.
    # Grand Total is intentionally kept out of the plot so the axis is not distorted by
    # one aggregated total bar when the user selects long periods such as LOM or 10Y.
    bench_totals = table.groupby("Bench", observed=True)["Au oz"].sum()
    max_bench_ounces = float(bench_totals.max()) if not bench_totals.empty else 0.0
    display_unit, display_divisor, text_format = _metal_at_risk_display_unit(max_bench_ounces)
    table["Au display"] = table["Au oz"] / display_divisor
    total_ounces = float(table["Au oz"].sum())
    total_display_value = total_ounces / display_divisor if display_divisor else total_ounces
    total_text = f"Total selected Au: {total_display_value:{text_format}} {display_unit}"

    category_order = ["Grade Control", "Measured", "Indicated (At Risk)"]

    fig = go.Figure()
    for category in category_order:
        group = table[table["Category"].eq(category)].copy()
        if group.empty:
            continue
        if use_numeric_bench_axis:
            group = group.sort_values("__bench_numeric__", ascending=False)
            x_values = group["__bench_numeric__"]
        else:
            group["__bench_order__"] = pd.Categorical(group["Bench"], categories=bench_labels, ordered=True)
            group = group.sort_values("__bench_order__")
            x_values = group["Bench"]
        fig.add_trace(
            go.Bar(
                x=x_values,
                y=group["Au display"],
                name=category,
                marker_color=METAL_AT_RISK_COLORS.get(category),
                marker_line_width=0.45,
                marker_line_color="white",
                text=group["Au display"],
                customdata=group["Au oz"],
                texttemplate=f"%{{text:{text_format}}}",
                textposition="auto",
                insidetextfont={"color": "white", "size": 10},
                outsidetextfont={"color": "#344B5A", "size": 10},
                hovertemplate=(
                    f"Bench: %{{x}}<br>%{{fullData.name}}: %{{y:{text_format}}} {display_unit}"
                    "<br>Au: %{customdata:,.0f} oz<extra></extra>"
                ),
            )
        )

    max_stack_display = float(table.groupby("Bench", observed=True)["Au display"].sum().max()) if not table.empty else 0.0
    y_range = [0, max_stack_display * 1.18] if max_stack_display > 0 else None

    fig.update_layout(
        title=f"Metal at Risk – {_active_year_scope_short_label()}",
        barmode="stack",
        height=520,
        legend_title_text="Category",
        legend={"orientation": "h", "y": -0.23, "x": 0.5, "xanchor": "center"},
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        margin={"l": 55, "r": 25, "t": 70, "b": 115},
        annotations=[
            {
                "text": total_text,
                "xref": "paper",
                "yref": "paper",
                "x": 1.0,
                "y": 1.09,
                "xanchor": "right",
                "yanchor": "top",
                "showarrow": False,
                "font": {"size": 12, "color": "#344B5A"},
                "bgcolor": "rgba(255,255,255,0.78)",
            }
        ],
    )
    if use_numeric_bench_axis:
        fig.update_xaxes(
            title_text="Bench",
            dtick=10,
            tickformat=".0f",
            tickangle=0,
            tickfont={"size": 9},
            autorange="reversed",
        )
    else:
        fig.update_xaxes(
            title_text="Bench",
            categoryorder="array",
            categoryarray=bench_labels,
            tickangle=0,
            tickfont={"size": 9},
        )
    fig.update_yaxes(
        title_text=f"Au ({display_unit})",
        range=y_range,
        showgrid=True,
        gridcolor="#D9D9D9",
        zeroline=True,
        zerolinecolor="#D9D9D9",
    )
    st.plotly_chart(fig, use_container_width=True)

def _plot_stacked_by_year(data: pd.DataFrame, config: ModelConfig, color_role: str, title: str, color_map: dict[str, str], percent: bool = True) -> pd.DataFrame:
    year_col = _year_column(config, data)
    color_col = config.column_for_role(color_role)
    if not year_col or not color_col or year_col not in data.columns or color_col not in data.columns or data.empty:
        st.info(f"Configure Year and {color_role} roles to activate this annual chart.")
        return pd.DataFrame()

    table = _group_tonnes(data, config, [year_col, color_col])
    if table.empty:
        return pd.DataFrame()

    ton_col = tonnage_column_name(config)
    year_label = config.category_label(year_col)
    color_label = config.category_label(color_col)
    table = table.rename(columns={year_col: year_label, color_col: color_label})
    table[year_label] = pd.to_numeric(table[year_label], errors="coerce")
    table = table.dropna(subset=[year_label]).copy()
    table[year_label] = table[year_label].astype(int)

    if color_role == "Category":
        table[color_label] = table[color_label].map(_display_resource_category)
        category_orders = {color_label: RESOURCE_CATEGORY_ORDER}
    elif color_role == "Destination":
        table[color_label] = table[color_label].map(_display_destination)
        category_orders = None
    else:
        table[color_label] = table[color_label].astype(str).str.strip().str.upper()
        category_orders = None

    fig = px.bar(
        table,
        x=year_label,
        y=ton_col,
        color=color_label,
        title=title,
        color_discrete_map=color_map,
        category_orders=category_orders,
    )
    if percent:
        fig.update_layout(barnorm="percent", yaxis_title="Distribution (%)")
    else:
        fig.update_layout(yaxis_title=ton_col)
    fig.update_layout(
        height=470,
        barmode="stack",
        legend_title_text=color_label,
        margin={"l": 25, "r": 20, "t": 65, "b": 45},
    )
    fig.update_xaxes(title_text="Year", dtick=1, tickformat=".0f")
    fig.update_traces(marker_line_width=0.5, marker_line_color="white")
    st.plotly_chart(fig, use_container_width=True)
    return table


def _render_year_distribution_section(
    data: pd.DataFrame,
    config: ModelConfig,
    destination_mode: str,
    filters: dict[str, Any] | None = None,
) -> None:
    year_col = _year_column(config, data)
    if not year_col or year_col not in data.columns:
        st.info("Configure a Year column to activate annual distribution plots and tables.")
        return

    st.markdown("#### Annual Distributions by Year")
    st.caption("These plots use the same master scope, sidebar filters and destination filter as the resource table above.")

    tabs = st.tabs(["Mettype by year", "Category by year", "Destination by year"])
    annual_specs = [
        ("Mettype", f"Met-Type Ore Distribution by Year ({destination_mode})", METTYPE_COLORS),
        ("Category", f"Resource Category Distribution by Year ({destination_mode})", RESOURCE_CATEGORY_COLORS),
        ("Destination", f"Destination Distribution by Year ({destination_mode})", DESTINATION_COLORS),
    ]

    for tab, (role, title, color_map) in zip(tabs, annual_specs, strict=True):
        with tab:
            table = _plot_stacked_by_year(data, config, role, title, color_map, percent=True)
            if not table.empty:
                st.markdown("##### Annual table")
                _render_barrick_table(table, config)
                _render_table_filter_note(
                    _table_filter_note_items(
                        config,
                        final_data=data,
                        filters=filters,
                        extra_items=[f"Annual grouping: {role} by Year"],
                    )
                )


def _plot_destination_by_bench_with_grade(data: pd.DataFrame, config: ModelConfig) -> None:
    bench_col = config.column_for_role("Bench")
    dest_col = config.column_for_role("Destination")
    if not bench_col or not dest_col or bench_col not in data.columns or dest_col not in data.columns or data.empty:
        st.info("Configure Bench and Destination roles to activate destination charts.")
        return

    table = _group_tonnes(data, config, [bench_col, dest_col])
    if table.empty:
        return
    ton_col = tonnage_column_name(config)
    bench_label = config.category_label(bench_col)
    dest_label = config.category_label(dest_col)
    table = table.rename(columns={bench_col: bench_label, dest_col: dest_label})
    table[dest_label] = table[dest_label].map(_display_destination)

    numeric_bench = pd.to_numeric(table[bench_label], errors="coerce")
    use_numeric_axis = numeric_bench.notna().all()
    if use_numeric_axis:
        table["__bench_x__"] = numeric_bench.astype(float)
        x_col = "__bench_x__"
    else:
        x_col = bench_label

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    secondary_axis_color = "#C08A00"
    secondary_grid_color = "rgba(192, 138, 0, 0.32)"
    for destination, group in table.groupby(dest_label, observed=True):
        fig.add_trace(
            go.Bar(
                x=group[x_col],
                y=group[ton_col],
                name=str(destination),
                marker_color=DESTINATION_COLORS.get(str(destination)),
                marker_line_width=0.5,
                marker_line_color="white",
            ),
            secondary_y=False,
        )

    au_spec = next((spec for spec in config.grade_specs if spec.label.casefold() == "au" and not _is_contained_metal_spec(spec)), None)
    if au_spec and au_spec.column in data.columns:
        grade_rows = []
        for bench, group in data.groupby(bench_col, observed=True):
            grade_rows.append({bench_label: bench, f"Au ({au_spec.unit})": weighted_mean(group[au_spec.column], group[config.mass_column])})
        grade_table = pd.DataFrame(grade_rows)
        if not grade_table.empty:
            if use_numeric_axis:
                grade_table["__bench_x__"] = pd.to_numeric(grade_table[bench_label], errors="coerce")
                grade_x = grade_table["__bench_x__"]
            else:
                grade_x = grade_table[bench_label]
            fig.add_trace(
                go.Scatter(
                    x=grade_x,
                    y=grade_table[f"Au ({au_spec.unit})"],
                    mode="lines+markers",
                    name=f"Au ({au_spec.unit})",
                    line={"color": secondary_axis_color, "width": 3},
                    marker={"color": secondary_axis_color, "size": 7},
                ),
                secondary_y=True,
            )

    fig.update_layout(
        title="Destination Ore Distribution by Bench",
        barmode="stack",
        height=500,
        legend_title_text=dest_label,
        margin={"l": 25, "r": 20, "t": 65, "b": 45},
    )
    if use_numeric_axis:
        fig.update_xaxes(title_text="BENCH", dtick=10, autorange="reversed", tickformat=".0f")
    else:
        fig.update_xaxes(title_text="BENCH", categoryorder="array", categoryarray=_bench_order(table[bench_label]))
    fig.update_yaxes(
        title_text=ton_col,
        secondary_y=False,
        showgrid=True,
        gridcolor="#DCE3EA",
        tickfont={"color": "#637083"},
        title_font={"color": "#637083"},
        linecolor="#8A96A3",
        tickcolor="#8A96A3",
        zerolinecolor="#B7C1CB",
    )
    fig.update_yaxes(
        title_text=f"Au ({au_spec.unit})" if au_spec else "Grade",
        secondary_y=True,
        showgrid=True,
        gridcolor=secondary_grid_color,
        tickfont={"color": secondary_axis_color},
        title_font={"color": secondary_axis_color},
        linecolor=secondary_axis_color,
        tickcolor=secondary_axis_color,
        zeroline=False,
    )
    # Plotly versions that support griddash will show a solid primary grid and
    # a dotted gold secondary grid. Older versions keep the differentiated colors.
    try:
        fig.update_yaxes(griddash="solid", secondary_y=False)
        fig.update_yaxes(griddash="dot", secondary_y=True)
    except Exception:
        pass
    st.plotly_chart(fig, use_container_width=True)

def _render_variable_controls(bundle: ModelBundle, model_name: str) -> None:
    config = bundle.config
    st.subheader("Variables and data controls")
    st.caption("Use this tab to confirm the model variables and define basic validation rules before reviewing the resource tabulation.")
    scoped_data = _apply_global_scope(bundle.data, config)
    _scope_caption(bundle.data, scoped_data, config)

    with st.form(f"controls_{_safe_key(model_name)}"):
        display_cols = st.columns(4)
        tonnage_unit = display_cols[0].selectbox(
            "Tonnage unit",
            TONNAGE_UNITS,
            index=TONNAGE_UNITS.index(config.tonnage_unit) if config.tonnage_unit in TONNAGE_UNITS else TONNAGE_UNITS.index("Mt"),
        )
        tonnage_decimals = int(display_cols[1].number_input("Tonnage decimals", 0, 6, int(config.tonnage_decimals)))
        grade_decimals = int(display_cols[2].number_input("Default grade decimals", 0, 6, int(config.grade_decimals)))
        report_title = display_cols[3].text_input("Report title", value=_display_report_title(config.report_title))

        validation_cols = st.columns(4)
        reject_negative_grades = validation_cols[0].checkbox(
            "Reject negative grades",
            value=bool(config.reject_negative_grades),
            help="When active, negative grade values are validation errors instead of warnings.",
        )
        require_positive_grades = validation_cols[1].checkbox(
            "Grades must be > 0",
            value=bool(config.require_positive_grades),
            help="When active, zero and negative grade values are validation errors.",
        )
        year_min = int(validation_cols[2].number_input("Minimum valid year", value=int(config.year_min)))
        year_max = int(validation_cols[3].number_input("Maximum valid year", value=int(config.year_max)))

        submitted = st.form_submit_button("Apply controls and revalidate", type="primary", use_container_width=True)

    if submitted:
        config.tonnage_unit = tonnage_unit
        config.tonnage_decimals = tonnage_decimals
        config.grade_decimals = grade_decimals
        config.report_title = report_title
        config.reject_negative_grades = reject_negative_grades
        config.require_positive_grades = require_positive_grades
        config.year_min = year_min
        config.year_max = year_max
        data, stats = clean_model_data(bundle.raw_data, config)
        validation = validate_model(data, config)
        st.session_state.models[model_name] = ModelBundle(config=config, raw_data=bundle.raw_data, data=data, validation=validation, cleaning_stats=stats)
        st.success("Controls applied and model revalidated.")
        st.rerun()

    st.markdown("#### Configured core variables")
    variable_rows = [
        {"Role": "Tonnage", "Column": config.mass_column, "Label": "Tonnage", "Control": "No nulls / no negatives"},
        {"Role": "Volume", "Column": config.volume_column or "N/A", "Label": "Volume", "Control": "No negatives"},
    ]
    variable_rows.extend(
        {"Role": spec.role, "Column": spec.column, "Label": spec.label, "Control": "Categorical filter/group"}
        for spec in config.category_specs
    )
    st.dataframe(_left_aligned_table_style(pd.DataFrame(variable_rows)), use_container_width=True, hide_index=True)
    _render_table_filter_note(
        _table_filter_note_items(
            config,
            source_data=bundle.data,
            final_data=scoped_data,
            model_name=model_name,
            extra_items=["This is a configuration table; no additional analytical filters are applied."],
        )
    )

    st.markdown("#### Configured grade variables")
    grade_rows = [
        {
            "Column": spec.column,
            "Label": spec.label,
            "Unit": spec.unit,
            "Decimals": spec.decimals,
            "Negative policy": "Error" if config.reject_negative_grades else "Warning",
            "Positive-only": "Yes" if config.require_positive_grades else "No",
        }
        for spec in config.grade_specs
    ]
    st.dataframe(_left_aligned_table_style(pd.DataFrame(grade_rows)), use_container_width=True, hide_index=True)
    _render_table_filter_note(
        _table_filter_note_items(
            config,
            source_data=bundle.data,
            final_data=scoped_data,
            model_name=model_name,
            extra_items=["This is a configured grade-variable list; no additional analytical filters are applied."],
        )
    )

    st.markdown("#### Negative/null/zero diagnostics")
    st.dataframe(_left_aligned_table_style(negative_counts(scoped_data, config)), use_container_width=True, hide_index=True)
    _render_table_filter_note(
        _table_filter_note_items(
            config,
            source_data=bundle.data,
            final_data=scoped_data,
            model_name=model_name,
            extra_items=["Diagnostics are calculated after master scope only; sidebar analytical filters are not applied in this controls tab."],
        )
    )


def _render_resource_dashboard(
    bundle: ModelBundle,
    model_name: str,
    base_data: pd.DataFrame | None = None,
    base_filters: dict[str, Any] | None = None,
) -> None:
    config = bundle.config
    st.subheader(_display_report_title(config.report_title))

    source_data = base_data.copy() if base_data is not None else bundle.data.copy()
    filtered = _apply_global_scope(source_data, config)
    filters = dict(base_filters or {})
    _scope_caption(source_data, filtered, config)

    destination_mode = _master_destination_label()
    # Destination / Ore Type is now a transversal master filter in the left sidebar.
    # Reapplying it here is idempotent and keeps this dashboard explicit and robust.
    filtered = _apply_destination_mode(filtered, config, destination_mode)

    if filtered.empty:
        st.warning("No rows match the sidebar filters and destination selection.")
        return

    _resource_metric_row(bundle, filtered)

    st.markdown(f"#### Tabulation by Categ ({destination_mode})")
    resource_table = _resource_tabulation(filtered, config)
    _render_barrick_table(resource_table, config)
    _render_table_filter_note(
        _table_filter_note_items(
            config,
            source_data=source_data,
            final_data=filtered,
            filters=filters,
            model_name=model_name,
            extra_items=["Resource-category table grouping: Measured, Indicated, Inferred, Inventory and Unclassified"],
        )
    )
    _add_scene_button(f"{model_name} - resource tabulation", "Resource tabulation", [model_name], resource_table, filters)

    with st.expander("Calculation audit / reconciliation inputs", expanded=False):
        st.caption(
            "The formulas are: weighted grade = sum(grade × tonnes) / sum(tonnes); "
            "Au (oz) and Ag (oz) = sum(g/t × tonnes) / 31.1034768; "
            "Cu (lb) = sum(Cu (%) / 100 × tonnes × 2,204.62262185)."
        )
        audit_table = _resource_calculation_audit(source_data, _apply_global_scope(source_data, config), filtered, config, destination_mode)
        _render_barrick_table(audit_table, config)
        _render_table_filter_note(
            _table_filter_note_items(
                config,
                source_data=source_data,
                final_data=filtered,
                filters=filters,
                model_name=model_name,
                extra_items=["Audit table reconciles uploaded rows, master scope and the selected destination mode"],
            )
        )

    st.markdown("#### Met-Type Ore Distribution by Bench")
    _plot_stacked_by_bench(filtered, config, "Mettype", f"Met-Type Ore Distribution by Bench ({destination_mode})", METTYPE_COLORS, percent=True)

    pie_cols = st.columns(2)
    with pie_cols[0]:
        _plot_mettype_distribution(filtered, config)
    with pie_cols[1]:
        _plot_role_distribution(filtered, config, "Destination", "Destination Ore Tonnes Distribution", DESTINATION_COLORS)

    st.markdown("#### Resource Category Distribution by Bench")
    _plot_stacked_by_bench(
        filtered,
        config,
        "Category",
        f"Drilling / Resource Category Distribution by Bench ({destination_mode})",
        RESOURCE_CATEGORY_COLORS,
        percent=True,
    )

    st.markdown("#### Metal at Risk")
    _plot_metal_at_risk_by_bench(filtered, config)

    st.markdown("#### Destination Ore Distribution by Bench")
    _plot_destination_by_bench_with_grade(filtered, config)

    cat_dest_cols = st.columns(2)
    with cat_dest_cols[0]:
        _plot_role_distribution(filtered, config, "Category", "Resource Category Tonnes Distribution", RESOURCE_CATEGORY_COLORS)
    with cat_dest_cols[1]:
        _plot_role_distribution(filtered, config, "Destination", "Destination Tonnes Distribution", DESTINATION_COLORS)

    st.markdown("#### Supporting Tables")
    table_tabs = st.tabs(["Mettype by bench", "Category by bench", "Destination by bench"])
    bench_col = config.column_for_role("Bench")
    for tab, role in zip(table_tabs, ["Mettype", "Category", "Destination"], strict=True):
        with tab:
            role_col = config.column_for_role(role)
            if bench_col and role_col:
                table = grouped_summary(filtered, config, [bench_col, role_col])
                _render_barrick_table(table, config)
                _render_table_filter_note(
                    _table_filter_note_items(
                        config,
                        source_data=source_data,
                        final_data=filtered,
                        filters=filters,
                        model_name=model_name,
                        extra_items=[f"Supporting table grouping: Bench by {role}"],
                    )
                )
            else:
                st.info(f"Bench and {role} roles are required.")

    _render_year_distribution_section(filtered, config, destination_mode, filters)


def _validation_status(error_count: int = 0, warning_count: int = 0) -> str:
    if error_count > 0:
        return "WARNING"
    if warning_count > 0:
        return "WARNING"
    return "OK"


def _zero_value_mask(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    text = series.astype("string").str.strip().str.lower()
    numeric_without_thousands = pd.to_numeric(text.str.replace(",", "", regex=False), errors="coerce")
    numeric_comma_decimal = pd.to_numeric(text.str.replace(",", ".", regex=False), errors="coerce")
    text_zero = text.isin(["zero", "zeros"])
    return (
        numeric.eq(0)
        | numeric_without_thousands.eq(0)
        | numeric_comma_decimal.eq(0)
        | text_zero
    ).fillna(False)


def _configured_column_role(config: ModelConfig, column: str) -> str:
    if column == config.mass_column:
        return "Mass"
    if column == config.volume_column:
        return "Volume"
    if any(spec.column == column for spec in config.grade_specs):
        return "Grade"
    for spec in config.category_specs:
        if spec.column == column:
            return spec.role
    return "Other"


def _valid_category_mask_for_validation(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    normalized = series.map(_normalize_label)
    text_valid = (
        (normalized.str.contains("grade", na=False) & normalized.str.contains("control", na=False))
        | normalized.str.contains("measured", na=False)
        | normalized.str.contains("indicated", na=False)
        | normalized.str.contains("inferred", na=False)
        | normalized.str.contains("inventory", na=False)
    )
    return numeric.isin([15, 1, 2, 3, 4]) | text_valid


def _column_validation_checklist(data: pd.DataFrame, config: ModelConfig) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    year_col = _year_column(config, data)
    bench_col = config.column_for_role("Bench")

    for column in data.columns:
        series = data[column]
        role = _configured_column_role(config, column)
        null_count = int(series.isna().sum())
        zero_mask = _zero_value_mask(series)
        zero_count = int(zero_mask.sum())
        warning_count = 1 if null_count else 0
        error_count = 0
        details: list[str] = []

        if null_count:
            details.append(f"{null_count:,} null values")

        if role in {"Mass", "Volume"}:
            numeric = pd.to_numeric(series, errors="coerce")
            non_numeric = int(series.notna().sum() - numeric.notna().sum())
            negative = int(numeric.lt(0).sum())
            error_count += int(non_numeric > 0) + int(negative > 0)
            details.append(f"numeric rows: {int(numeric.notna().sum()):,}")
            if non_numeric:
                details.append(f"non-numeric: {non_numeric:,}")
            zero = int(numeric.eq(0).sum())
            if zero:
                details.append(f"zero values accepted: {zero:,}")
            if negative:
                details.append(f"negative values: {negative:,}")

        elif column == year_col:
            numeric = pd.to_numeric(series, errors="coerce")
            non_numeric = int(series.notna().sum() - numeric.notna().sum())
            non_positive = int(numeric.le(0).sum())
            out_of_setup_range = int(((numeric < config.year_min) | (numeric > config.year_max)).sum())
            error_count += int(non_numeric > 0) + int(non_positive > 0) + int(out_of_setup_range > 0)
            if numeric.notna().any():
                details.append(f"loaded range: {int(numeric.min())}-{int(numeric.max())}")
            if non_numeric:
                details.append(f"non-numeric: {non_numeric:,}")
            if non_positive:
                details.append(f"<= 0 years: {non_positive:,}")
            if out_of_setup_range:
                details.append(f"outside setup range {config.year_min}-{config.year_max}: {out_of_setup_range:,}")

        elif column == bench_col:
            numeric = pd.to_numeric(series, errors="coerce")
            non_numeric = int(series.notna().sum() - numeric.notna().sum())
            outside_range = int(((numeric < -500) | (numeric > 500)).sum())
            error_count += int(non_numeric > 0) + int(outside_range > 0)
            if numeric.notna().any():
                details.append(f"loaded range: {numeric.min():,.0f} to {numeric.max():,.0f}")
            if non_numeric:
                details.append(f"non-numeric: {non_numeric:,}")
            if outside_range:
                details.append(f"outside -500 to 500: {outside_range:,}")

        elif role == "Grade":
            numeric = pd.to_numeric(series, errors="coerce")
            non_numeric = int(series.notna().sum() - numeric.notna().sum())
            negative = int(numeric.lt(0).sum())
            zero = int(numeric.eq(0).sum())
            positive = int(numeric.gt(0).sum())
            error_count += int(non_numeric > 0) + int(negative > 0) + int(zero > 0)
            details.append(f"positive values used for stats: {positive:,}")
            if negative:
                details.append(f"negative values: {negative:,}")
            if zero:
                details.append(f"zero values: {zero:,}")
            if non_numeric:
                details.append(f"non-numeric: {non_numeric:,}")

        elif role == "Category":
            numeric = pd.to_numeric(series, errors="coerce")
            valid = _valid_category_mask_for_validation(series)
            zero_values = zero_count
            invalid = series.notna() & ~valid & ~zero_mask
            minus_99 = int(numeric.eq(-99).sum())
            invalid_count = int(invalid.sum())
            error_count += int(invalid_count > 0) + int(minus_99 > 0)
            warning_count += int(zero_values > 0)
            details.append("valid codes: 15, 1, 2, 3, 4")
            if invalid_count:
                invalid_values = sorted(series[invalid].dropna().astype(str).unique().tolist())[:8]
                details.append(f"invalid values: {', '.join(invalid_values)}")
            if zero_values:
                details.append(f"zero values: {zero_values:,}")
            if minus_99:
                details.append(f"-99 values: {minus_99:,}")

        rows.append(
            {
                "Status": _validation_status(error_count, warning_count),
                "Column": column,
                "Role": role,
                "Rows": len(series),
                "Nulls": null_count,
                "Zero Values": zero_count,
                "Details": "; ".join(details) if details else "No issues detected",
            }
        )

    return pd.DataFrame(rows)


def _grade_positive_statistics(data: pd.DataFrame, config: ModelConfig) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for spec in config.grade_specs:
        if _is_contained_metal_spec(spec) or spec.column not in data.columns:
            continue

        numeric = pd.to_numeric(data[spec.column], errors="coerce")
        positive_mask = numeric.gt(0)
        positive = numeric[positive_mask]

        rows.append(
            {
                "Column": spec.column,
                "Display name": _canonical_grade_label(spec),
                "Unit": _effective_grade_unit(spec),
                "Positive values": int(positive_mask.sum()),
                "Zero values": int(numeric.eq(0).sum()),
                "Negative values": int(numeric.lt(0).sum()),
                "Non-numeric": int(data[spec.column].notna().sum() - numeric.notna().sum()),
                "Min (>0)": float(positive.min()) if not positive.empty else float("nan"),
                "Mean (>0)": float(positive.mean()) if not positive.empty else float("nan"),
                "Max (>0)": float(positive.max()) if not positive.empty else float("nan"),
            }
        )

    return pd.DataFrame(rows)


def _validation_control_summary(data: pd.DataFrame, config: ModelConfig) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    year_col = _year_column(config, data)
    bench_col = config.column_for_role("Bench")
    category_col = config.column_for_role("Category")

    if year_col and year_col in data.columns:
        year = pd.to_numeric(data[year_col], errors="coerce")
        rows.append(
            {
                "Control": "Year",
                "Column": year_col,
                "Valid records": int(year.gt(0).sum()),
                "Flagged records": int((year.le(0) | year.isna() | (year < config.year_min) | (year > config.year_max)).sum()),
                "Loaded range": f"{int(year.min())}-{int(year.max())}" if year.notna().any() else "N/A",
                "Rule": f"numeric, positive, setup range {config.year_min}-{config.year_max}",
            }
        )

    if bench_col and bench_col in data.columns:
        bench = pd.to_numeric(data[bench_col], errors="coerce")
        rows.append(
            {
                "Control": "Bench",
                "Column": bench_col,
                "Valid records": int((bench.notna() & bench.between(-500, 500)).sum()),
                "Flagged records": int((bench.isna() | (bench < -500) | (bench > 500)).sum()),
                "Loaded range": f"{bench.min():,.0f} to {bench.max():,.0f}" if bench.notna().any() else "N/A",
                "Rule": "numeric; values from -500 to 500 are accepted",
            }
        )

    if category_col and category_col in data.columns:
        category = data[category_col]
        valid = _valid_category_mask_for_validation(category)
        rows.append(
            {
                "Control": "Resource category",
                "Column": category_col,
                "Valid records": int(valid.sum()),
                "Flagged records": int((category.notna() & ~valid).sum()),
                "Loaded range": ", ".join(sorted(category.dropna().astype(str).unique().tolist())[:12]),
                "Rule": "accepted values: 15 Grade Control, 1 Measured, 2 Indicated, 3 Inferred, 4 Inventory",
            }
        )

    return pd.DataFrame(rows)


def _histogram_stats_text(values: pd.Series) -> str:
    if values.empty:
        return "n: 0"
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    cv = std / mean if mean else float("nan")
    return "<br>".join(
        [
            f"n: {len(values):,}",
            f"m: {mean:,.3f}",
            f"s: {std:,.3f}",
            f"CV: {cv:,.2f}" if pd.notna(cv) else "CV: -",
            f"min: {float(values.min()):,.3f}",
            f"P10: {float(values.quantile(0.10)):,.3f}",
            f"P50: {float(values.quantile(0.50)):,.3f}",
            f"P90: {float(values.quantile(0.90)):,.3f}",
            f"max: {float(values.max()):,.3f}",
        ]
    )


def _render_grade_distribution_multiplot(data: pd.DataFrame, config: ModelConfig, model_name: str) -> None:
    available_specs = [
        spec
        for spec in config.grade_specs
        if not _is_contained_metal_spec(spec) and spec.column in data.columns
    ]
    if not available_specs:
        return

    options = [spec.column for spec in available_specs]
    default = options[: min(4, len(options))]
    selected = st.multiselect(
        "Distribution variables",
        options,
        default=default,
        key=f"validation_dist_multiplot_{_safe_key(model_name)}",
        help="Select up to four grade variables. Histograms use positive values only.",
    )
    if not selected:
        st.info("Select at least one grade variable to display the distribution multiplot.")
        return
    if len(selected) > 4:
        st.warning("Only the first four selected variables are shown in the multiplot.")
        selected = selected[:4]

    flagged_rows: list[dict[str, Any]] = []
    plot_values: dict[str, pd.Series] = {}
    for column in selected[:4]:
        numeric = pd.to_numeric(data[column], errors="coerce")
        positive = numeric[numeric.gt(0)].dropna()
        plot_values[column] = positive
        flagged_rows.append(
            {
                "Variable": column,
                "Positive values plotted": int(positive.count()),
                "Zero values excluded": int(numeric.eq(0).sum()),
                "Negative values excluded": int(numeric.lt(0).sum()),
                "Non-numeric excluded": int(data[column].notna().sum() - numeric.notna().sum()),
            }
        )

    flagged = pd.DataFrame(flagged_rows)
    excluded_total = int(
        flagged[["Zero values excluded", "Negative values excluded", "Non-numeric excluded"]].sum().sum()
    )
    if excluded_total:
        st.warning(f"{excluded_total:,} zero, negative or non-numeric values were excluded from the selected distributions.")
    if all(values.empty for values in plot_values.values()):
        st.info("No positive values are available for the selected variables.")
        with st.expander("Distribution exclusion summary", expanded=True):
            st.dataframe(flagged, use_container_width=True, hide_index=True)
            _render_table_filter_note(
                _table_filter_note_items(
                    config,
                    final_data=data,
                    model_name=model_name,
                    extra_items=["Distribution exclusion summary uses the selected distribution variables and positive-value rule."],
                )
            )
        return

    rows, cols = 2, 2
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=[column for column in selected[:4]],
        horizontal_spacing=0.08,
        vertical_spacing=0.16,
    )

    for index, column in enumerate(selected[:4]):
        row = index // cols + 1
        col = index % cols + 1
        values = plot_values[column]
        if values.empty:
            continue

        fig.add_trace(
            go.Histogram(
                x=values,
                nbinsx=35,
                marker={
                    "color": "#0068C9",
                    "line": {"color": "#FFFFFF", "width": 1.2},
                },
                hovertemplate=f"{column}<br>Value: %{{x}}<br>Count: %{{y}}<extra></extra>",
                showlegend=False,
            ),
            row=row,
            col=col,
        )
        fig.add_annotation(
            text=_histogram_stats_text(values),
            xref=f"x{index + 1 if index else ''} domain",
            yref=f"y{index + 1 if index else ''} domain",
            x=0.98,
            y=0.96,
            xanchor="right",
            yanchor="top",
            align="left",
            showarrow=False,
            font={"size": 10, "color": "#1F2933"},
            bgcolor="rgba(255,255,255,0.78)",
            bordercolor="rgba(0,0,0,0.12)",
            borderwidth=1,
        )
        fig.update_xaxes(title_text=column, showgrid=True, gridcolor="#D9E2EC", row=row, col=col)
        fig.update_yaxes(title_text="Count", showgrid=True, gridcolor="#D9E2EC", row=row, col=col)

    fig.update_layout(
        title="Grade Distribution Multiplot (> 0 only)",
        height=720,
        bargap=0.02,
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        margin={"l": 45, "r": 35, "t": 75, "b": 45},
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Distribution exclusion summary", expanded=False):
        st.dataframe(flagged, use_container_width=True, hide_index=True)
        _render_table_filter_note(
            _table_filter_note_items(
                config,
                final_data=data,
                model_name=model_name,
                extra_items=["Distribution exclusion summary uses the selected distribution variables and positive-value rule."],
            )
        )


def _render_validation_details(bundle: ModelBundle, model_name: str) -> None:
    config = bundle.config
    st.subheader("Validation details")
    _metric_row(bundle, bundle.data)

    st.markdown("#### Validation checklist")
    checklist = _column_validation_checklist(bundle.data, config)
    st.dataframe(
        checklist,
        use_container_width=True,
        hide_index=True,
        column_config=_text_table_column_config(
            checklist,
            narrow_columns=("Status", "Role", "Rows", "Nulls", "Zero Values"),
            medium_columns=("Column",),
            wide_columns=("Details",),
        ),
    )
    _render_table_filter_note(
        _table_filter_note_items(
            config,
            final_data=bundle.data,
            model_name=model_name,
            extra_items=["Validation details use the master-scoped model; sidebar analytical filters are not applied."],
        )
    )

    issues = bundle.validation.as_frame()
    if issues.empty:
        st.success("No validation issues were detected by the configured validation engine.")
    else:
        st.markdown("#### Validation engine issues")
        st.dataframe(
            issues,
            use_container_width=True,
            hide_index=True,
            column_config=_text_table_column_config(issues),
        )
        _render_table_filter_note(
            _table_filter_note_items(
                config,
                final_data=bundle.data,
                model_name=model_name,
                extra_items=["Validation issues are calculated after master scope only."],
            )
        )

    control_summary = _validation_control_summary(bundle.data, config)
    if not control_summary.empty:
        st.markdown("#### Year, bench and category controls")
        st.dataframe(
            control_summary,
            use_container_width=True,
            hide_index=True,
            column_config=_text_table_column_config(
                control_summary,
                narrow_columns=("Control", "Valid records", "Flagged records"),
                medium_columns=("Column", "Loaded range"),
                wide_columns=("Rule",),
            ),
        )
        _render_table_filter_note(
            _table_filter_note_items(
                config,
                final_data=bundle.data,
                model_name=model_name,
                extra_items=["Control summary is calculated after master scope only."],
            )
        )

    grade_stats = _grade_positive_statistics(bundle.data, config)
    if not grade_stats.empty:
        st.markdown("#### Grade statistics (positive values only)")
        st.caption("Zero and negative grade values are excluded from these statistics and reported in the checklist.")
        st.dataframe(
            _left_aligned_table_style(
                grade_stats,
                formatters=_table_number_formatters(grade_stats, 3),
                na_rep="-",
            ),
            use_container_width=True,
            hide_index=True,
        )
        _render_table_filter_note(
            _table_filter_note_items(
                config,
                final_data=bundle.data,
                model_name=model_name,
                extra_items=["Grade statistics use positive grade values only after master scope."],
            )
        )

    if config.grade_specs:
        st.markdown("#### Grade distribution multiplot")
        _render_grade_distribution_multiplot(bundle.data, config, model_name)


# -----------------------------------------------------------------------------
# Pages
# -----------------------------------------------------------------------------


def _setup_step(step: int, title: str, description: str) -> None:
    """Render a consistent visual landmark without changing setup behavior."""
    st.markdown(
        f"""
        <section class="bm-setup-step">
            <div class="bm-setup-step-number">{step}</div>
            <div>
                <div class="bm-setup-step-title">{html.escape(title)}</div>
                <div class="bm-setup-step-description">{html.escape(description)}</div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def _render_no_models_state(message: str, minimum: int = 1) -> None:
    """Consistent empty state for analytical pages."""
    noun = "model" if minimum == 1 else "models"
    st.markdown(
        f"""
        <section class="bm-empty-state bm-analytical-empty">
            <div class="bm-empty-state-icon">{minimum:02d}</div>
            <div>
                <div class="bm-empty-state-title">{html.escape(message)}</div>
                <div class="bm-empty-state-text">Configure at least {minimum} {noun} to activate this workspace.</div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Go to Model Setup", type="primary", key=f"empty_setup_{_safe_key(message)}"):
        move_to("Model Setup")


def _render_active_model_strip(model_name: str, bundle: ModelBundle) -> None:
    """Display existing model/session facts without changing analytical scope."""
    validation_label = "Review required" if bundle.validation.is_blocked else "Ready"
    st.markdown(
        f"""
        <section class="bm-model-context">
            <div><span>Active model</span><strong>{html.escape(model_name)}</strong></div>
            <div><span>Source rows</span><strong>{len(bundle.data):,}</strong></div>
            <div><span>Grades</span><strong>{len(bundle.config.grade_specs):,}</strong></div>
            <div><span>Validation</span><strong>{validation_label}</strong></div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def _render_volume_gate_state(passed: bool, message: str, model_count: int, tolerance: float) -> None:
    state_class = "is-passed" if passed else "is-review"
    state_label = "Passed" if passed else "Review required"
    st.markdown(
        f"""
        <section class="bm-gate-state {state_class}" role="status">
            <div class="bm-gate-state-mark">{"✓" if passed else "!"}</div>
            <div class="bm-gate-state-copy">
                <div class="bm-gate-state-label">Global volume gate · {state_label}</div>
                <div class="bm-gate-state-message">{html.escape(message)}</div>
            </div>
            <div class="bm-gate-state-meta">
                <span>{model_count} models</span>
                <span>{tolerance:.4f}% tolerance</span>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_home() -> None:
    st.markdown(
        f"""
        <section class="bm-home-masthead">
            <div>
                <div class="bm-home-masthead-kicker">Mineral Resource Management</div>
                <div class="bm-home-masthead-title">BlockModel Studio</div>
                <div class="bm-home-masthead-subtitle">Resource tabulation and model-comparison workspace for mineral-resource block models.</div>
            </div>
            <div class="bm-home-masthead-brand">
                <div class="bm-home-masthead-logo">{_barrick_logo_html(128)}</div>
                <div class="bm-home-masthead-brand-label">MRM</div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    vulcan_res_path = ROOT / "assets" / "BlockModel_Studio_Vulcan_Model_Query.res"
    st.markdown(
        """
        <section class="bm-vulcan-resource">
            <div class="bm-vulcan-resource-copy">
                <div class="bm-vulcan-resource-kicker">Vulcan utility</div>
                <div class="bm-vulcan-resource-title">Model query specification</div>
                <div class="bm-vulcan-resource-text">
                    Download the prepared <strong>.res</strong> specification when you need to query a
                    block model in Maptek Vulcan before loading the resulting tabulation here.
                    Review the model variables and region paths in Vulcan before running it.
                </div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )
    if vulcan_res_path.exists():
        st.download_button(
            "Download Vulcan query (.res)",
            data=vulcan_res_path.read_bytes(),
            file_name="BlockModel_Studio_Vulcan_Model_Query.res",
            mime="application/octet-stream",
            key="download_vulcan_model_query_res",
            help="Save the specification and run it from Maptek Vulcan after checking variables and region paths.",
        )
    else:
        st.error("The embedded Vulcan query specification is unavailable.")

    home_actions = st.columns(3)
    if home_actions[0].button("Configure a model", type="primary", use_container_width=True):
        move_to("Model Setup")
    if home_actions[1].button("Open evaluation", use_container_width=True, disabled=not st.session_state.models):
        move_to("Model Evaluation")
    if home_actions[2].button("Compare models", use_container_width=True, disabled=len(st.session_state.models) < 2):
        move_to("Model Comparison")

    components.html(
        """
        <style>
        body {
            margin: 0;
            background: transparent;
            font-family: "Source Sans Pro", Arial, sans-serif;
        }
        .pv-home-wrap {
            margin: 0;
            border-radius: 10px;
            overflow: hidden;
            border: none;
            box-sizing: border-box;
            background: linear-gradient(135deg, #003B5C 0%, #03547C 48%, #F7F9FB 48%, #FFFFFF 100%);
            box-shadow: none;
        }
        .pv-home-hero {
            width: 100%;
            height: 318px;
            display: block;
        }
        </style>
        <div class="pv-home-wrap">
          <svg class="pv-home-hero" viewBox="0 0 1500 360" preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Mineral resource model evaluation, bench distribution and model comparison">
            <defs>
              <linearGradient id="pvBlue" x1="0" x2="1" y1="0" y2="1">
                <stop offset="0" stop-color="#003B5C"/>
                <stop offset="1" stop-color="#03547C"/>
              </linearGradient>
              <linearGradient id="panelFill" x1="0" x2="1" y1="0" y2="1">
                <stop offset="0" stop-color="#F8FBFD"/>
                <stop offset="1" stop-color="#EAF2F7"/>
              </linearGradient>
              <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">
                <feDropShadow dx="0" dy="12" stdDeviation="14" flood-color="#003B5C" flood-opacity="0.16"/>
              </filter>
            </defs>

            <rect x="0" y="0" width="1500" height="360" fill="#FFFFFF"/>
            <rect x="0" y="0" width="590" height="360" fill="url(#pvBlue)"/>
            <path d="M505 0 L710 0 L585 360 L0 360 L0 0 Z" fill="#004967" opacity="0.70"/>
            <path d="M0 286 C132 238 260 312 395 264 C468 238 525 198 610 220 L575 360 L0 360 Z" fill="#A39161" opacity="0.20"/>

            <g transform="matrix(1.18 0 0 1.18 -42 -20)">
              <rect x="58" y="34" width="445" height="286" rx="18" fill="rgba(255,255,255,0.08)" stroke="rgba(255,255,255,0.22)" stroke-width="2"/>

              <g transform="translate(0,-28)">
              <g opacity="0.24">
                <line x1="92" y1="262" x2="462" y2="262" stroke="#FFFFFF" stroke-width="2"/>
                <line x1="92" y1="118" x2="92" y2="262" stroke="#FFFFFF" stroke-width="2"/>
                <line x1="92" y1="226" x2="462" y2="226" stroke="#FFFFFF" stroke-width="1"/>
                <line x1="92" y1="190" x2="462" y2="190" stroke="#FFFFFF" stroke-width="1"/>
                <line x1="92" y1="154" x2="462" y2="154" stroke="#FFFFFF" stroke-width="1"/>
                <text x="68" y="266" fill="#FFFFFF" font-size="10">100</text>
                <text x="68" y="230" fill="#FFFFFF" font-size="10">150</text>
                <text x="68" y="194" fill="#FFFFFF" font-size="10">200</text>
                <text x="68" y="158" fill="#FFFFFF" font-size="10">250</text>
              </g>

              <path d="M92 226 C148 188 196 206 246 174 C290 146 332 148 386 124 C420 108 446 116 462 110 L462 262 L92 262 Z" fill="#0E6A8B" opacity="0.56"/>
              <path d="M92 262 L92 224 C164 218 205 206 250 180 C303 204 348 194 462 154 L462 262 Z" fill="#189653" opacity="0.55"/>
              <path d="M194 262 C214 224 239 196 280 176 C311 162 346 151 385 128 L462 111 L462 262 Z" fill="#A39161" opacity="0.34"/>

              <g opacity="0.94">
                <rect x="126" y="222" width="26" height="18" fill="#70AD47" stroke="#FFFFFF" stroke-width="0.7"/>
                <rect x="154" y="211" width="26" height="18" fill="#70AD47" stroke="#FFFFFF" stroke-width="0.7"/>
                <rect x="182" y="199" width="26" height="18" fill="#5B9BD5" stroke="#FFFFFF" stroke-width="0.7"/>
                <rect x="210" y="188" width="26" height="18" fill="#70AD47" stroke="#FFFFFF" stroke-width="0.7"/>
                <rect x="238" y="177" width="26" height="18" fill="#ED7D31" stroke="#FFFFFF" stroke-width="0.7"/>
                <rect x="266" y="166" width="26" height="18" fill="#ED7D31" stroke="#FFFFFF" stroke-width="0.7"/>
                <rect x="294" y="156" width="26" height="18" fill="#A39161" stroke="#FFFFFF" stroke-width="0.7"/>
                <rect x="322" y="145" width="26" height="18" fill="#5B9BD5" stroke="#FFFFFF" stroke-width="0.7"/>
                <rect x="350" y="135" width="26" height="18" fill="#A39161" stroke="#FFFFFF" stroke-width="0.7"/>
                <rect x="378" y="126" width="26" height="18" fill="#ED7D31" stroke="#FFFFFF" stroke-width="0.7"/>
                <rect x="154" y="231" width="26" height="18" fill="#189653" stroke="#FFFFFF" stroke-width="0.7"/>
                <rect x="182" y="219" width="26" height="18" fill="#70AD47" stroke="#FFFFFF" stroke-width="0.7"/>
                <rect x="210" y="208" width="26" height="18" fill="#5B9BD5" stroke="#FFFFFF" stroke-width="0.7"/>
                <rect x="238" y="197" width="26" height="18" fill="#ED7D31" stroke="#FFFFFF" stroke-width="0.7"/>
                <rect x="266" y="186" width="26" height="18" fill="#A39161" stroke="#FFFFFF" stroke-width="0.7"/>
                <rect x="294" y="176" width="26" height="18" fill="#A39161" stroke="#FFFFFF" stroke-width="0.7"/>
                <rect x="322" y="165" width="26" height="18" fill="#5B9BD5" stroke="#FFFFFF" stroke-width="0.7"/>
                <rect x="350" y="155" width="26" height="18" fill="#70AD47" stroke="#FFFFFF" stroke-width="0.7"/>
                <rect x="378" y="146" width="26" height="18" fill="#ED7D31" stroke="#FFFFFF" stroke-width="0.7"/>
              </g>

              <ellipse cx="292" cy="177" rx="84" ry="32" fill="none" stroke="#FFFFFF" stroke-width="3" stroke-dasharray="8 7" opacity="0.82" transform="rotate(-22 292 177)"/>
              <text x="356" y="104" fill="#FFFFFF" opacity="0.90" font-size="11" font-weight="700">Search ellipsoid</text>
              <line x1="352" y1="108" x2="326" y2="145" stroke="#FFFFFF" stroke-width="1.5" opacity="0.72"/>

              <g stroke-linecap="round" opacity="0.95">
                <line x1="132" y1="108" x2="194" y2="250" stroke="#FFFFFF" stroke-width="3"/>
                <line x1="132" y1="108" x2="194" y2="250" stroke="#64C7F3" stroke-width="1.4" stroke-dasharray="4 7"/>
                <circle cx="148" cy="145" r="4" fill="#A39161"/>
                <circle cx="162" cy="178" r="4" fill="#A39161"/>
                <circle cx="176" cy="211" r="4" fill="#A39161"/>
                <text x="106" y="105" fill="#FFFFFF" opacity="0.88" font-size="11" font-weight="700">Drillholes</text>

                <line x1="245" y1="105" x2="284" y2="251" stroke="#FFFFFF" stroke-width="3"/>
                <line x1="245" y1="105" x2="284" y2="251" stroke="#64C7F3" stroke-width="1.4" stroke-dasharray="4 7"/>
                <circle cx="254" cy="139" r="4" fill="#A39161"/>
                <circle cx="264" cy="177" r="4" fill="#A39161"/>
                <circle cx="274" cy="215" r="4" fill="#A39161"/>

                <line x1="394" y1="96" x2="342" y2="249" stroke="#FFFFFF" stroke-width="3"/>
                <line x1="394" y1="96" x2="342" y2="249" stroke="#64C7F3" stroke-width="1.4" stroke-dasharray="4 7"/>
                <circle cx="382" cy="132" r="4" fill="#A39161"/>
                <circle cx="370" cy="168" r="4" fill="#A39161"/>
                <circle cx="358" cy="205" r="4" fill="#A39161"/>
              </g>

              <g opacity="0.70" fill="none" stroke-width="2">
                <path d="M108 298 C160 282 210 304 262 287 C318 269 358 294 433 272" stroke="#FFFFFF"/>
                <path d="M108 306 C160 292 217 314 270 298 C320 282 364 305 433 285" stroke="#A39161"/>
                <path d="M108 314 C160 303 218 322 274 309 C322 298 371 316 433 301" stroke="#64C7F3"/>
              </g>
              <g transform="translate(78,278)">
                <rect x="0" y="0" width="14" height="10" fill="#70AD47"/>
                <text x="20" y="10" fill="#FFFFFF" opacity="0.82" font-size="10">low</text>
                <rect x="66" y="0" width="14" height="10" fill="#A39161"/>
                <text x="86" y="10" fill="#FFFFFF" opacity="0.82" font-size="10">mid</text>
                <rect x="130" y="0" width="14" height="10" fill="#ED7D31"/>
                <text x="150" y="10" fill="#FFFFFF" opacity="0.82" font-size="10">high Au</text>
              </g>
              </g>
            </g>

            <g filter="url(#shadow)">
              <rect x="640" y="34" rx="18" ry="18" width="790" height="286" fill="url(#panelFill)" stroke="#D9E4EC"/>

              <g transform="translate(704,58)">
                <g transform="translate(20,18)">
                  <polygon points="0,44 86,18 86,74 0,100" fill="#176D8F"/>
                  <polygon points="90,17 155,0 155,57 90,74" fill="#189653"/>
                  <polygon points="160,0 220,20 220,78 160,57" fill="#8C8C8C"/>
                  <polygon points="225,22 292,2 292,65 225,80" fill="#FF1F1F"/>
                  <polygon points="32,105 105,84 105,142 32,165" fill="#169653"/>
                  <polygon points="110,84 180,60 180,122 110,142" fill="#0A5A7C"/>
                  <polygon points="185,60 258,82 258,145 185,122" fill="#3B36FF"/>
                  <polygon points="263,84 330,64 330,126 263,145" fill="#A39161"/>
                  <polygon points="80,170 150,148 150,210 80,230" fill="#FF1F1F"/>
                  <polygon points="155,148 225,124 225,188 155,210" fill="#189653"/>
                  <polygon points="230,124 302,146 302,208 230,188" fill="#1F6D8C"/>
                </g>
              </g>

              <g transform="translate(1040,52)">
                <rect x="0" y="0" width="275" height="148" rx="14" fill="#FFFFFF" stroke="#D6E2EB" stroke-width="2"/>
                <rect x="24" y="28" width="210" height="8" fill="#03547C"/>
                <line x1="24" y1="62" x2="238" y2="62" stroke="#DDE6EE" stroke-width="4"/>
                <line x1="24" y1="92" x2="238" y2="92" stroke="#DDE6EE" stroke-width="4"/>
                <line x1="24" y1="122" x2="182" y2="122" stroke="#DDE6EE" stroke-width="4"/>
              </g>

              <g transform="translate(703,236)">
                <line x1="0" y1="95" x2="585" y2="95" stroke="#CBD8E2"/>
                <rect x="20" y="42" width="24" height="53" fill="#70AD47"/>
                <rect x="48" y="12" width="24" height="83" fill="#808080"/>
                <rect x="102" y="30" width="24" height="65" fill="#70AD47"/>
                <rect x="130" y="52" width="24" height="43" fill="#808080"/>
                <rect x="184" y="18" width="24" height="77" fill="#70AD47"/>
                <rect x="212" y="36" width="24" height="59" fill="#808080"/>
                <rect x="266" y="48" width="24" height="47" fill="#70AD47"/>
                <rect x="294" y="22" width="24" height="73" fill="#808080"/>
                <rect x="348" y="28" width="24" height="67" fill="#70AD47"/>
                <rect x="376" y="58" width="24" height="37" fill="#808080"/>
                <rect x="430" y="14" width="24" height="81" fill="#70AD47"/>
                <rect x="458" y="34" width="24" height="61" fill="#808080"/>
                <rect x="512" y="44" width="24" height="51" fill="#70AD47"/>
                <rect x="540" y="24" width="24" height="71" fill="#808080"/>
              </g>

              <g transform="translate(1087,204)">
                <polyline points="0,92 82,48 160,66 240,18" fill="none" stroke="#03547C" stroke-width="5"/>
                <circle cx="0" cy="92" r="7" fill="#A39161"/>
                <circle cx="82" cy="48" r="7" fill="#A39161"/>
                <circle cx="160" cy="66" r="7" fill="#A39161"/>
                <circle cx="240" cy="18" r="7" fill="#A39161"/>
                <rect x="0" y="122" width="74" height="16" rx="8" fill="#5B9BD5"/>
                <rect x="82" y="122" width="74" height="16" rx="8" fill="#70AD47"/>
                <rect x="164" y="122" width="74" height="16" rx="8" fill="#ED7D31"/>
              </g>
            </g>
          </svg>
        </div>
        """,
        height=326,
        scrolling=False,
    )

    if st.session_state.models:
        _render_master_year_filter_sidebar(list(st.session_state.models.values()), "home")
        _render_master_phase_filter_sidebar(list(st.session_state.models.values()), "home")
        _render_master_destination_filter_sidebar(list(st.session_state.models.values()), "home")

    st.markdown(
        """
        <div class="pv-home-info-grid">
            <div class="pv-home-info-card">
                <div class="pv-home-info-title">Model evaluation</div>
                <div class="pv-home-info-text">Validate one block model, review resource tabulation, and analyze distributions by bench, category, destination and mettype.</div>
            </div>
            <div class="pv-home-info-card">
                <div class="pv-home-info-title">Model comparison</div>
                <div class="pv-home-info-text">Compare multiple configured models in the same workspace, including the mandatory global-volume compatibility check.</div>
            </div>
            <div class="pv-home-info-card">
                <div class="pv-home-info-title">Large tabulations</div>
                <div class="pv-home-info-text">Designed for large CSV or Excel block-model tabulations; practical limits depend on local memory and file size.</div>
            </div>
            <div class="pv-home-info-card">
                <div class="pv-home-info-title">Transversal filters</div>
                <div class="pv-home-info-text">Master filters for MetType, Phase, Bench and Year scope are applied consistently across evaluation, comparison and reports.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Current Workspace")
    if not st.session_state.models:
        st.markdown(
            """
            <section class="bm-empty-state">
                <div class="bm-empty-state-icon">BM</div>
                <div>
                    <div class="bm-empty-state-title">Your workspace is ready</div>
                    <div class="bm-empty-state-text">Upload a CSV, TXT or Excel block-model tabulation in Model Setup to begin an evaluation.</div>
                </div>
            </section>
            """,
            unsafe_allow_html=True,
        )
    else:
        rows = []
        for name, bundle in st.session_state.models.items():
            scoped = _scoped_bundle(bundle)
            rows.append(
                {
                    "Model": name,
                    "Type": bundle.config.model_type,
                    "Rows after scope": len(scoped.data),
                    "Raw rows": len(bundle.data),
                    "Grades": len(bundle.config.grade_specs),
                    "Categories": len(bundle.config.category_specs),
                    "Validation": "Blocked" if scoped.validation.is_blocked else "Ready",
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

def render_setup() -> None:
    page_header("Model Setup", "Load a block-model tabulation and map the variables required for analysis.")

    _setup_step(1, "Define the global model scope", "Choose which BLK_MODEL records will be available throughout the workspace. Value 0 is always excluded.")
    master_options = list(MASTER_BLK_MODEL_OPTIONS)
    current_master = st.session_state.get("master_blk_model_scope", "1 - In situ")
    setup_master_key = "setup_master_blk_model_scope_selector"
    if st.session_state.get(setup_master_key) != current_master:
        st.session_state[setup_master_key] = current_master
    setup_master = st.selectbox(
        "Default model scope",
        options=master_options,
        index=master_options.index(current_master) if current_master in master_options else 0,
        key=setup_master_key,
        help="1 = in situ, 2 = stockpiles. Select 1+2 only when you intentionally want both block-model scopes.",
    )
    if setup_master != current_master:
        st.session_state.master_blk_model_scope = setup_master
        st.rerun()

    saved_notice = st.session_state.pop("setup_model_saved_notice", None)
    if saved_notice:
        saved_name = str(saved_notice.get("name", "Model"))
        if bool(saved_notice.get("blocked", False)):
            st.warning(
                f"Model '{saved_name}' was loaded and listed below, but it has critical validation issues that require review."
            )
        else:
            st.success(
                f"Model '{saved_name}' was loaded successfully and is ready for evaluation or comparison."
            )
        st.caption("The upload control has been reset and is ready for the next model.")

    _render_configured_models_manager("setup")

    _setup_step(2, "Load a model tabulation", "Upload the CSV, TXT or Excel output produced from your block-model query.")
    uploader_version = int(st.session_state.get("setup_uploader_version", 0))
    uploaded = st.file_uploader(
        "CSV or Excel model tabulation",
        type=["csv", "txt", "xlsx", "xlsm", "xls"],
        key=f"setup_model_uploader_{uploader_version}",
    )
    if uploaded is not None:
        signature = f"{uploaded.name}:{uploaded.size}"
        if st.session_state.setup_filename != signature:
            try:
                st.session_state.setup_raw = load_dataframe(uploaded.name, uploaded.getvalue())
                st.session_state.setup_filename = signature
                st.success(f"Loaded {uploaded.name}")
            except Exception as exc:
                st.error(f"Could not read source file: {exc}")

    frame = st.session_state.setup_raw
    if frame is None:
        st.markdown(
            """
            <section class="bm-empty-state bm-empty-state-compact">
                <div class="bm-empty-state-icon">02</div>
                <div>
                    <div class="bm-empty-state-title">Waiting for a source tabulation</div>
                    <div class="bm-empty-state-text">Accepted formats: CSV, TXT, XLSX, XLSM and XLS. The file remains in the current Streamlit session.</div>
                </div>
            </section>
            """,
            unsafe_allow_html=True,
        )
        return

    source_summary = st.columns(3)
    source_summary[0].metric("Source rows", f"{len(frame):,}")
    source_summary[1].metric("Source columns", f"{len(frame.columns):,}")
    source_summary[2].metric("File status", "Loaded")
    st.caption(f"Source: {st.session_state.setup_filename}")
    with st.expander("Preview source data", expanded=False):
        st.dataframe(frame.head(100), use_container_width=True, hide_index=True)

    columns = list(frame.columns)
    _setup_step(3, "Map model variables", "Confirm identity, mass and volume fields, grades, categories, cleaning rules and display units.")
    with st.form("model_configuration"):
        st.markdown("#### Model identity")
        identity = st.columns([1.1, 1.0, 1.7])
        default_name = Path(str(st.session_state.setup_filename)).stem.replace("-", "_").replace(" ", "_")[:45]
        model_name = identity[0].text_input("Model name", value=default_name)
        model_type = identity[1].selectbox("Model type", MODEL_TYPES)
        report_title = identity[2].text_input("Report title", value=_display_report_title(f"{model_name} Evaluation", "Block Model Evaluation"))

        st.markdown("#### Core variables")
        core = st.columns(4)
        mass_index = _default_index(columns, column_aliases().get("mass_column_aliases", ["TONNES", "TOTAL_MASS", "MASS", "TONNAGE"]))
        mass_column = core[0].selectbox("Tonnage / mass column", columns, index=mass_index)
        volume_options = ["None"] + columns
        volume_index = _default_index(volume_options, column_aliases().get("volume_column_aliases", ["VOLUME", "TOTAL_VOLUME", "TOTAL_VOLUMEN"]), fallback=0)
        volume_selection = core[1].selectbox("Volume column", volume_options, index=volume_index)
        volume_column = None if volume_selection == "None" else volume_selection
        default_volume_unit = "Mm3"
        volume_unit_index = VOLUME_UNITS.index(default_volume_unit) if default_volume_unit in VOLUME_UNITS else 0
        volume_unit = core[2].selectbox("Volume unit", VOLUME_UNITS, index=volume_unit_index)
        default_tonnage_unit = display_defaults().get("default_tonnage_unit", "Mt")
        tonnage_unit = core[3].selectbox("Display tonnage unit", TONNAGE_UNITS, index=TONNAGE_UNITS.index(default_tonnage_unit) if default_tonnage_unit in TONNAGE_UNITS else TONNAGE_UNITS.index("Mt"))

        st.markdown("#### Grade variables")
        grade_options = [
            column for column in columns
            if column not in {mass_column, volume_column}
            and _is_grade_like_column(column)
        ]
        grade_columns = st.multiselect("Select up to nine grade variables", grade_options, default=_suggest_grades(grade_options))
        max_grades = _configured_max_grades()
        if len(grade_columns) > max_grades:
            st.error(f"Select a maximum of {max_grades} grade variables.")
        grade_specs: list[GradeSpec] = []
        for index, column in enumerate(grade_columns[:max_grades], start=1):
            row = st.columns([1.2, 1.2, 0.8, 0.8])
            row[0].text_input("Column", value=column, disabled=True, key=f"grade_source_{index}_{_safe_key(column)}")
            label = row[1].text_input("Display name", value=_default_grade_label(column), key=f"grade_label_{index}_{_safe_key(column)}")
            default_unit = _default_grade_unit(column)
            unit = row[2].selectbox("Unit", GRADE_UNITS, index=GRADE_UNITS.index(default_unit), key=f"grade_unit_{index}_{_safe_key(column)}")
            decimals = int(row[3].number_input("Decimals", min_value=0, max_value=6, value=int(display_defaults().get("grade_decimals", 2)), key=f"grade_dec_{index}_{_safe_key(column)}"))
            grade_specs.append(GradeSpec(column=column, label=label, unit=unit, decimals=decimals))

        st.markdown("#### Categorical filters and grouping")
        category_options = [
            column for column in columns
            if column not in {mass_column, volume_column, *grade_columns}
            and _is_category_candidate(frame, column)
        ]
        category_columns = st.multiselect("Select category/filter columns", category_options, default=_suggest_categories(frame))
        category_specs: list[CategorySpec] = []
        for index, column in enumerate(category_columns, start=1):
            row = st.columns([1.3, 0.9])
            label = row[0].text_input("Display name", value=column, key=f"cat_label_{index}_{_safe_key(column)}")
            inferred = _infer_role(column)
            role = row[1].selectbox("Role", CATEGORY_ROLES, index=CATEGORY_ROLES.index(inferred), key=f"cat_role_{index}_{_safe_key(column)}")
            category_specs.append(CategorySpec(column=column, label=label, role=role))

        st.markdown("#### Cleaning and display preferences")
        prefs = st.columns(5)
        null_text = prefs[0].text_input("Null tokens", value=",".join(cleaning_defaults().get("null_tokens", ["NA", "N/A", "NULL", "None", "-99", "-999", "-9999"])))
        default_decimal_separator = cleaning_defaults().get("decimal_separator", ".")
        decimal_separator = prefs[1].selectbox("Decimal separator", [".", ","], index=0 if default_decimal_separator == "." else 1)
        default_thousands_separator = cleaning_defaults().get("thousands_separator", ",")
        thousands_options = [",", ".", "None"]
        thousands_separator = prefs[2].selectbox("Thousands separator", thousands_options, index=thousands_options.index(default_thousands_separator) if default_thousands_separator in thousands_options else 0)
        tonnage_decimals = int(prefs[3].number_input("Tonnage decimals", min_value=0, max_value=6, value=int(display_defaults().get("tonnage_decimals", 2))))
        grade_decimals = int(prefs[4].number_input("Default grade decimals", min_value=0, max_value=6, value=int(display_defaults().get("grade_decimals", 2))))
        year_limits = st.columns(2)
        year_min = int(year_limits[0].number_input("Minimum valid year", value=int(validation_defaults().get("year_min", 1900))))
        year_max = int(year_limits[1].number_input("Maximum valid year", value=int(validation_defaults().get("year_max", 2200))))

        _setup_step(4, "Validate and save", "Create the configured model in this workspace and make it available to Evaluation, Comparison and Reports.")
        submitted = st.form_submit_button("Validate and save model", type="primary", use_container_width=True)

    if submitted:
        if not model_name.strip():
            st.error("Model name is required.")
            return
        if not grade_specs:
            st.warning("No grade variables were selected. The app can still validate mass/volume/category data.")
        config = ModelConfig(
            model_name=model_name.strip(),
            model_type=model_type,
            report_title=report_title,
            mass_column=mass_column,
            volume_column=volume_column,
            volume_unit=volume_unit,
            grade_specs=grade_specs,
            category_specs=category_specs,
            null_tokens=[token.strip() for token in null_text.split(",")],
            decimal_separator=decimal_separator,
            thousands_separator="" if thousands_separator == "None" else thousands_separator,
            tonnage_unit=tonnage_unit,
            tonnage_decimals=tonnage_decimals,
            grade_decimals=grade_decimals,
            year_min=year_min,
            year_max=year_max,
        )
        data, stats = clean_model_data(frame, config)
        validation = validate_model(data, config)
        st.session_state.models[model_name.strip()] = ModelBundle(
            config=config,
            raw_data=frame.copy(),
            data=data,
            validation=validation,
            cleaning_stats=stats,
        )
        st.session_state.setup_model_saved_notice = {
            "name": model_name.strip(),
            "blocked": bool(validation.is_blocked),
        }
        st.session_state.setup_raw = None
        st.session_state.setup_filename = ""
        st.session_state.setup_uploader_version = uploader_version + 1
        st.rerun()

def _series_unique_values(series: pd.Series, max_items: int = 14, natural_sort: bool = False) -> tuple[int, str]:
    clean = series.dropna().astype(str).str.strip()
    clean = clean[clean.ne("")]
    if clean.empty:
        return 0, "N/A"

    values = clean.unique().tolist()
    if natural_sort:
        values = sorted(values, key=_natural_phase_sort_key)
    else:
        values = sorted(values, key=lambda item: str(item).casefold())

    preview = values[:max_items]
    suffix = f", +{len(values) - max_items} more" if len(values) > max_items else ""
    return len(values), ", ".join(map(str, preview)) + suffix


def _series_numeric_range(series: pd.Series, decimals: int = 0) -> str:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return "N/A"
    if decimals <= 0:
        return f"{numeric.min():,.0f} to {numeric.max():,.0f}"
    return f"{numeric.min():,.{decimals}f} to {numeric.max():,.{decimals}f}"


def _configured_role_column(config: ModelConfig, data: pd.DataFrame, role: str) -> str | None:
    if role in {"Phase", "Pit_Phase"}:
        return _phase_column(config, data)
    if role == "Year":
        return _year_column(config, data)
    if role == "Block Model":
        return _block_model_column(config, data)
    column = config.column_for_role(role)
    return column if column and column in data.columns else None


def _categorical_role_summary_rows(data: pd.DataFrame, config: ModelConfig) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    roles_to_report = [
        "Mettype",
        "Phase",
        "Pit_Phase",
        "Pit",
        "Year",
        "Bench",
        "Category",
        "Destination",
        "Domain",
        "Lithology",
        "Alteration",
        "Weathering",
        "Block Model",
        "Source",
    ]

    seen_columns: set[str] = set()
    for role in roles_to_report:
        column = _configured_role_column(config, data, role)
        if not column or column not in data.columns or column in seen_columns:
            continue
        seen_columns.add(column)
        count, values = _series_unique_values(data[column], natural_sort=role in {"Phase", "Pit_Phase"})
        details = ""
        if role == "Year":
            details = f"Range: {_series_numeric_range(data[column])}"
        elif role == "Bench":
            details = f"Range: {_series_numeric_range(data[column])}"
        elif role in {"Phase", "Pit_Phase"}:
            details = "Unified Phase/Pit_Phase field"

        rows.append(
            {
                "Role": "Phase / Pit_Phase" if role in {"Phase", "Pit_Phase"} else role,
                "Column": column,
                "Unique values": count,
                "Values / range": values if role not in {"Year", "Bench"} else _series_numeric_range(data[column]),
                "Details": details or "Configured categorical/filter variable",
            }
        )

    for spec in config.category_specs:
        if spec.column not in data.columns or spec.column in seen_columns:
            continue
        count, values = _series_unique_values(data[spec.column])
        rows.append(
            {
                "Role": spec.role,
                "Column": spec.column,
                "Unique values": count,
                "Values / range": values,
                "Details": "Additional configured category/filter variable",
            }
        )
        seen_columns.add(spec.column)

    return rows


def _model_description_rows(model_name: str, raw_bundle: ModelBundle, scoped_bundle: ModelBundle) -> list[dict[str, Any]]:
    config = raw_bundle.config
    data = scoped_bundle.data
    raw_data = raw_bundle.data
    rows: list[dict[str, Any]] = []

    def add(metric: str, value: Any, detail: str = "") -> None:
        rows.append({"Metric": metric, "Value": value, "Detail": detail})

    add("Model name", model_name, config.model_type)
    add("Report title", _display_report_title(config.report_title), "Shown in evaluation outputs")
    add("Raw records", f"{len(raw_data):,}", "Before master filters")
    add("Records after master scope", f"{len(data):,}", f"{len(data) / len(raw_data) * 100:,.1f}% retained" if len(raw_data) else "")
    add("Columns", f"{len(raw_data.columns):,}", "Source-model columns")
    add("Configured grade variables", f"{len(config.grade_specs):,}", ", ".join(f"{spec.label} ({spec.unit})" for spec in config.grade_specs) or "None")
    add("Configured categorical filters", f"{len(config.category_specs):,}", ", ".join(f"{spec.role}: {spec.column}" for spec in config.category_specs) or "None")

    if config.mass_column in data.columns:
        tonnes = total_tonnage(data, config) / tonnage_divisor(config.tonnage_unit)
        add(f"Total tonnes ({config.tonnage_unit})", f"{tonnes:,.{config.tonnage_decimals}f}", f"Mass column: {config.mass_column}")

    if config.volume_column and config.volume_column in data.columns:
        display_volume, volume_label = _display_volume_total(data, config)
        add(f"Total volume ({volume_label})", _format_volume_value(display_volume), f"Volume column: {config.volume_column}")

    year_col = _year_column(config, data)
    if year_col and year_col in data.columns:
        count, _ = _series_unique_values(data[year_col])
        add("Years", f"{count:,}", f"Range: {_series_numeric_range(data[year_col])}; column: {year_col}")

    phase_col = _phase_column(config, data)
    if phase_col and phase_col in data.columns:
        count, values = _series_unique_values(data[phase_col], natural_sort=True)
        add("Mining phases", f"{count:,}", f"{values}; column: {phase_col}")

    bench_col = config.column_for_role("Bench")
    if bench_col and bench_col in data.columns:
        count, _ = _series_unique_values(data[bench_col], natural_sort=True)
        add("Benches", f"{count:,}", f"Range: {_series_numeric_range(data[bench_col])}; column: {bench_col}")

    for role in ["Mettype", "Category", "Destination", "Pit", "Domain", "Lithology", "Alteration", "Weathering"]:
        column = config.column_for_role(role)
        if column and column in data.columns:
            count, values = _series_unique_values(data[column], natural_sort=role == "Pit")
            add(role, f"{count:,}", f"{values}; column: {column}")

    blk_col = _block_model_column(config, data)
    if blk_col and blk_col in data.columns:
        count, values = _series_unique_values(data[blk_col])
        add("Block model scope values", f"{count:,}", f"{values}; column: {blk_col}")

    return rows


def _model_description_overview(scoped_bundles: dict[str, ModelBundle]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_name, bundle in scoped_bundles.items():
        config = bundle.config
        year_col = _year_column(config, bundle.data)
        phase_col = _phase_column(config, bundle.data)
        bench_col = config.column_for_role("Bench")
        mettype_col = config.column_for_role("Mettype")
        categ_col = config.column_for_role("Category")
        destination_col = config.column_for_role("Destination")

        def unique_count(column: str | None) -> int | str:
            if not column or column not in bundle.data.columns:
                return "N/A"
            count, _ = _series_unique_values(bundle.data[column])
            return count

        rows.append(
            {
                "Model": model_name,
                "Rows after scope": len(bundle.data),
                f"Tonnes ({config.tonnage_unit})": total_tonnage(bundle.data, config) / tonnage_divisor(config.tonnage_unit),
                "Mettypes": unique_count(mettype_col),
                "Phases": unique_count(phase_col),
                "Years": unique_count(year_col),
                "Year range": _series_numeric_range(bundle.data[year_col]) if year_col and year_col in bundle.data.columns else "N/A",
                "Benches": unique_count(bench_col),
                "Bench range": _series_numeric_range(bundle.data[bench_col]) if bench_col and bench_col in bundle.data.columns else "N/A",
                "Categories": unique_count(categ_col),
                "Destinations": unique_count(destination_col),
                "Grades": len(config.grade_specs),
            }
        )

    return pd.DataFrame(rows)


def _blank_value_count(series: pd.Series) -> int:
    text = series.astype("string").str.strip()
    return int(text.isna().sum() + text.eq("").sum())


def _role_completeness_rows(model_name: str, bundle: ModelBundle) -> list[dict[str, Any]]:
    config = bundle.config
    data = bundle.data
    rows: list[dict[str, Any]] = []

    def add(role: str, column: str | None, group: str = "Categorical") -> None:
        if not column or column not in data.columns:
            rows.append(
                {
                    "Model": model_name,
                    "Group": group,
                    "Role": role,
                    "Column": "Not configured",
                    "Rows": len(data),
                    "Non-null / non-blank": 0,
                    "Missing / blank": len(data),
                    "Completeness (%)": 0.0 if len(data) else float("nan"),
                    "Zero values": "N/A",
                    "Unique values": 0,
                }
            )
            return

        series = data[column]
        blank = _blank_value_count(series)
        non_blank = int(len(series) - blank)
        numeric = pd.to_numeric(series, errors="coerce")
        zero_values: int | str = int(numeric.eq(0).sum()) if numeric.notna().any() else "N/A"
        unique_count, _ = _series_unique_values(series)
        rows.append(
            {
                "Model": model_name,
                "Group": group,
                "Role": role,
                "Column": column,
                "Rows": len(series),
                "Non-null / non-blank": non_blank,
                "Missing / blank": blank,
                "Completeness (%)": (non_blank / len(series) * 100.0) if len(series) else float("nan"),
                "Zero values": zero_values,
                "Unique values": unique_count,
            }
        )

    add("Tonnage", config.mass_column, "Core")
    add("Volume", config.volume_column, "Core")

    for role in [
        "Mettype",
        "Phase / Pit_Phase",
        "Pit",
        "Year",
        "Bench",
        "Category",
        "Destination",
        "Domain",
        "Lithology",
        "Alteration",
        "Weathering",
        "Block Model",
    ]:
        if role == "Phase / Pit_Phase":
            column = _phase_column(config, data)
        elif role == "Year":
            column = _year_column(config, data)
        elif role == "Block Model":
            column = _block_model_column(config, data)
        else:
            column = config.column_for_role(role)
        add(role, column, "Categorical")

    return rows


def _model_description_recommended_checks(scoped_bundles: dict[str, ModelBundle]) -> None:
    st.markdown("#### Additional Model Description/Checks")
    st.caption("Completeness by role highlights whether the core and categorical fields required for evaluation/comparison are populated after the active master scope.")

    rows: list[dict[str, Any]] = []
    for model_name, bundle in scoped_bundles.items():
        rows.extend(_role_completeness_rows(model_name, bundle))

    if not rows:
        st.info("No configured models are available for completeness checks.")
        return

    completeness_table = pd.DataFrame(rows)
    st.dataframe(
        completeness_table.style.format({"Completeness (%)": "{:,.1f}"}, na_rep="N/A"),
        use_container_width=True,
        hide_index=True,
    )
    st.caption("Completeness (%) = non-null/non-blank records divided by rows after the active master BLK_MODEL, Year and Phase/Pit_Phase scope.")


def render_model_description() -> None:
    page_header("Model description", "Summarize configured model structure, categorical filters and scope before evaluation or comparison.")
    if not st.session_state.models:
        _render_no_models_state("No models available for description")
        return

    raw_bundles = list(st.session_state.models.values())
    _render_master_year_filter_sidebar(raw_bundles, "description")
    _render_master_phase_filter_sidebar(raw_bundles, "description")
    _render_master_destination_filter_sidebar(raw_bundles, "description")

    scoped_bundles = {name: _scoped_bundle(bundle) for name, bundle in st.session_state.models.items()}

    st.markdown("### Model overview")
    overview = _model_description_overview(scoped_bundles)
    st.dataframe(
        overview.style.format(_table_number_formatters(overview, 3), na_rep="N/A"),
        use_container_width=True,
        hide_index=True,
    )

    model_names = list(st.session_state.models)
    tabs = st.tabs(model_names)
    for tab, model_name in zip(tabs, model_names, strict=True):
        with tab:
            raw_bundle = st.session_state.models[model_name]
            scoped_bundle = scoped_bundles[model_name]
            _scope_caption(raw_bundle.data, scoped_bundle.data, scoped_bundle.config)

            st.markdown("#### Model description")
            description_table = pd.DataFrame(_model_description_rows(model_name, raw_bundle, scoped_bundle))
            st.dataframe(
                description_table,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Metric": st.column_config.TextColumn(width=270),
                    "Value": st.column_config.TextColumn(width=205),
                    "Detail": st.column_config.TextColumn(width=790),
                },
            )

            st.markdown("#### Configured categorical/filter variables")
            role_rows = _categorical_role_summary_rows(scoped_bundle.data, scoped_bundle.config)
            if role_rows:
                role_table = pd.DataFrame(role_rows)
                st.dataframe(
                    role_table,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Role": st.column_config.TextColumn(width=170),
                        "Column": st.column_config.TextColumn(width=190),
                        "Unique values": st.column_config.TextColumn(width=140),
                        "Values / range": st.column_config.TextColumn(width=285),
                        "Details": st.column_config.TextColumn(width=530),
                    },
                )
            else:
                st.info("No categorical/filter variables have been configured for this model.")

            st.markdown("#### Grade-variable inventory")
            grade_rows = [
                {
                    "Column": spec.column,
                    "Display name": spec.label,
                    "Unit": spec.unit,
                    "Decimals": spec.decimals,
                    "Available in data": "Yes" if spec.column in scoped_bundle.data.columns else "No",
                }
                for spec in scoped_bundle.config.grade_specs
            ]
            if grade_rows:
                # Keep this descriptive inventory fully left-aligned, including
                # the Decimals column, by rendering all values as text.
                grade_table = pd.DataFrame(grade_rows).astype(str)
                st.dataframe(
                    _left_aligned_table_style(grade_table),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Column": st.column_config.TextColumn(width=210),
                        "Display name": st.column_config.TextColumn(width=235),
                        "Unit": st.column_config.TextColumn(width=180),
                        "Decimals": st.column_config.TextColumn(width=190),
                        "Available in data": st.column_config.TextColumn(width=250),
                    },
                )
            else:
                st.info("No grade variables were configured for this model.")

    with st.expander("Additional Model Description/Checks", expanded=False):
        _model_description_recommended_checks(scoped_bundles)


def render_quality() -> None:
    """Backward-compatible route: old Data Quality page now renders Model Description."""
    render_model_description()


def render_evaluation() -> None:
    _apply_premium_tab_styles()
    page_header("Model Evaluation", "Configure data controls and review the resource tabulation dashboard for one model.")
    if not st.session_state.models:
        _render_no_models_state("No model available for evaluation")
        return

    model_name = st.selectbox("Model to evaluate", list(st.session_state.models), key="eval_model")
    bundle = st.session_state.models[model_name]
    _render_active_model_strip(model_name, bundle)
    _render_master_year_filter_sidebar([bundle], f"eval_{_safe_key(model_name)}")
    _render_master_phase_filter_sidebar([bundle], f"eval_{_safe_key(model_name)}")
    _render_master_destination_filter_sidebar([bundle], f"eval_{_safe_key(model_name)}")
    scoped_bundle = _scoped_bundle(bundle)

    filtered_data, sidebar_filters = _sidebar_evaluation_filters(scoped_bundle, f"eval_{_safe_key(model_name)}")

    tabs = st.tabs([
        "Variables & controls",
        "Validation details",
        "Tabulation by Categ",
        "Tabulation by Destination",
    ])

    with tabs[0]:
        _render_variable_controls(bundle, model_name)

    with tabs[1]:
        st.caption("Validation details are shown after the master BLK_MODEL, Year, Phase and Destination / Ore Type scope. Sidebar analytical filters are intended for evaluation charts and tables.")
        _render_validation_details(scoped_bundle, model_name)

    with tabs[2]:
        _render_resource_dashboard(scoped_bundle, model_name, filtered_data, sidebar_filters)

    with tabs[3]:
        _render_resource_by_destination(scoped_bundle, model_name, filtered_data, sidebar_filters)

def render_comparison() -> None:
    _apply_premium_tab_styles()
    page_header("Model Comparison", "Compare two or more configured models after the mandatory global-volume check.")
    if len(st.session_state.models) < 2:
        _render_no_models_state("Two models are required for comparison", minimum=2)
        return

    selected = st.multiselect("Models to compare", list(st.session_state.models), default=list(st.session_state.models)[:2])
    if len(selected) < 2:
        st.warning("Select at least two models.")
        return

    selected_raw_bundles = [st.session_state.models[name] for name in selected]
    _render_master_year_filter_sidebar(selected_raw_bundles, "comparison")
    _render_master_phase_filter_sidebar(selected_raw_bundles, "comparison")
    _render_master_destination_filter_sidebar(selected_raw_bundles, "comparison")
    bundles = {name: _scoped_bundle(st.session_state.models[name]) for name in selected}

    tolerance = st.number_input(
        "Global volume tolerance (%)",
        min_value=0.0,
        max_value=10.0,
        value=float(validation_defaults().get("global_volume_tolerance_pct", 0.10)),
        step=0.01,
        format="%.4f",
    )
    volume_table = compare_global_volumes(bundles)
    passed, message = volume_gate_status(volume_table, tolerance)

    _render_volume_gate_state(passed, message, len(selected), tolerance)
    accept_override = st.checkbox("I accept continuing even if the global-volume validation does not pass.", value=False)

    tabs = st.tabs(["Global volume check", "Tabulation by Destination"])

    with tabs[0]:
        st.subheader("Global-Volume Validation")
        with st.expander("Master scope applied to comparison", expanded=False):
            for name in selected:
                original = st.session_state.models[name]
                scoped = bundles[name]
                _scope_caption(original.data, scoped.data, scoped.config)
        volume_display_table = _scale_volume_columns_for_display(volume_table, bundles)
        st.dataframe(
            _left_aligned_table_style(
                volume_display_table,
                formatters=_table_number_formatters(volume_display_table, 4),
                na_rep="N/A",
            ),
            use_container_width=True,
            hide_index=True,
        )
        _render_table_filter_note(
            _comparison_filter_note_items(
                bundles,
                selected,
                extra_items=[f"Global volume tolerance: {tolerance:.4f}%", f"Override accepted: {'Yes' if accept_override else 'No'}"],
            )
        )
        matrix = pairwise_volume_matrix(bundles)
        st.caption("Pairwise variance in %, using each row as reference model.")
        st.dataframe(
            _left_aligned_table_style(
                matrix,
                formatters=_table_number_formatters(matrix, 4),
                na_rep="N/A",
            ),
            use_container_width=True,
            hide_index=True,
        )
        _render_table_filter_note(
            _comparison_filter_note_items(
                bundles,
                selected,
                extra_items=[f"Pairwise variance table uses each row model as the reference; tolerance shown above = {tolerance:.4f}%"],
            )
        )

    with tabs[1]:
        if not passed and not accept_override:
            st.info("Destination comparison is locked until the volume check passes or you explicitly accept the override.")
            return
        _render_comparison_resource_by_destination(bundles, selected)


# -----------------------------------------------------------------------------
# Automatic reporting helpers
# -----------------------------------------------------------------------------


def _report_model_label(model_names: list[str]) -> str:
    if not model_names:
        return "No configured models"
    if len(model_names) == 1:
        return model_names[0]
    if len(model_names) <= 3:
        return ", ".join(model_names)
    return ", ".join(model_names[:3]) + f", +{len(model_names) - 3} more"


def _report_file_slug(text_value: str, fallback: str = "block_model_report") -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", str(text_value or fallback)).strip("_").lower()
    return slug[:70] or fallback


def _report_destination_mode_for_model(model_name: str) -> str:
    # Kept for backward compatibility with the reporting code; destination scope is now global.
    return _master_destination_label()


def _report_table_sheet_name(base: str, used: set[str]) -> str:
    clean = re.sub(r"[\\/*?:\[\]]", "_", str(base)).strip() or "Sheet"
    clean = clean[:31]
    candidate = clean
    index = 2
    while candidate in used:
        suffix = f"_{index}"
        candidate = clean[: 31 - len(suffix)] + suffix
        index += 1
    used.add(candidate)
    return candidate


def _report_table_item(title: str, table: pd.DataFrame, notes: list[str] | None = None, category: str = "Table") -> dict[str, Any]:
    return {"title": title, "table": table.copy() if isinstance(table, pd.DataFrame) else pd.DataFrame(), "notes": notes or [], "category": category}


def _notes_frame(notes: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"Notes": [f"{index}. {note}" for index, note in enumerate(notes, start=1)]})


def _automatic_report_tables(scoped_bundles: dict[str, ModelBundle]) -> list[dict[str, Any]]:
    """Collect only the principal evaluation and comparison result tables."""
    items: list[dict[str, Any]] = []
    if not scoped_bundles:
        return items

    for model_name, bundle in scoped_bundles.items():
        raw_bundle = st.session_state.models.get(model_name, bundle)
        config = bundle.config
        destination_mode = _report_destination_mode_for_model(model_name)
        resource_data = _apply_destination_mode(bundle.data, config, destination_mode)
        resource_filters: dict[str, Any] = {}

        resource_table = _resource_tabulation(resource_data, config)
        if not resource_table.empty:
            items.append(_report_table_item(
                f"Tabulation by Categ - {model_name}",
                resource_table,
                _table_filter_note_items(
                    config,
                    source_data=raw_bundle.data,
                    final_data=resource_data,
                    filters=resource_filters,
                    model_name=model_name,
                    extra_items=["Resource-category grouping: Grade Control, Measured, Indicated, Inferred, Inventory and Unclassified."],
                ),
                "Model Evaluation",
            ))

        destination_table = _destination_summary_table(bundle.data, config, model_name).drop(columns=["__row_type__"], errors="ignore")
        if not destination_table.empty:
            items.append(_report_table_item(
                f"Tabulation by Destination - {model_name}",
                destination_table,
                _table_filter_note_items(
                    config,
                    source_data=raw_bundle.data,
                    final_data=bundle.data,
                    model_name=model_name,
                    extra_items=["Destination grouping includes ore, waste, grades, contained metal and resource-category tonnes."],
                ),
                "Model Evaluation",
            ))

    if len(scoped_bundles) >= 2:
        selected_names = list(scoped_bundles.keys())
        reference_model = st.session_state.get("comparison_destination_reference_model", selected_names[0])
        if reference_model not in selected_names:
            reference_model = selected_names[0]
        comp_destination = _destination_comparison_table(scoped_bundles, reference_model).drop(columns=["__row_type__"], errors="ignore")
        if not comp_destination.empty:
            items.append(_report_table_item(
                "Model Comparison - Tabulation by Destination",
                comp_destination,
                _comparison_filter_note_items(
                    scoped_bundles,
                    selected_names,
                    reference_model,
                    extra_items=["Vertical comparison table by model, including relative differences against the selected reference model."],
                ),
                "Model Comparison",
            ))

    return items

def _is_report_text_column(column_name: Any) -> bool:
    text = str(column_name).casefold()
    keywords = [
        "note", "comment", "description", "issue", "message", "filter", "scope",
        "table", "section", "model", "column", "role", "category", "destination",
        "mettype", "phase", "pit", "lithology", "alteration", "weathering",
    ]
    return any(keyword in text for keyword in keywords)


def _is_report_numeric_series(series: pd.Series) -> bool:
    if series.empty:
        return False
    numeric = pd.to_numeric(series, errors="coerce")
    valid = series.notna() & series.astype("string").str.strip().ne("")
    if int(valid.sum()) == 0:
        return False
    return float(numeric.notna().sum()) / float(valid.sum()) >= 0.82


def _excel_column_width(table: pd.DataFrame, column: Any) -> int:
    """Compact content-driven Excel width; long narrative fields wrap instead of stretching the sheet."""
    name = str(column)
    values = table[column].head(500).tolist() if column in table.columns else []
    lengths = [len(name)] + [len(str(value).replace("\n", " ")) for value in values if value is not None]
    lengths = sorted(lengths)
    representative = lengths[min(len(lengths) - 1, int(round((len(lengths) - 1) * 0.90)))] if lengths else len(name)
    if _is_report_text_column(name):
        return int(min(max(representative + 3, 16), 42))
    if column in table.columns and _is_report_numeric_series(table[column]):
        return int(min(max(representative + 2, 9), 16))
    return int(min(max(representative + 2, 10), 24))

def _report_excel_bytes(items: list[dict[str, Any]]) -> bytes:
    output = io.BytesIO()
    used_names: set[str] = set()
    try:
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            workbook = writer.book
            title_fmt = workbook.add_format({"bold": True, "font_color": "#03547C", "font_size": 12, "align": "center", "valign": "vcenter"})
            header_fmt = workbook.add_format({
                "bold": True,
                "bg_color": "#03547C",
                "font_color": "#FFFFFF",
                "border": 1,
                "align": "center",
                "valign": "vcenter",
                "text_wrap": True,
            })
            body_fmt = workbook.add_format({"border": 1, "valign": "vcenter", "align": "center"})
            body_text_fmt = workbook.add_format({"border": 1, "valign": "vcenter", "align": "center", "text_wrap": True})
            body_num_fmt = workbook.add_format({"border": 1, "valign": "vcenter", "align": "center", "num_format": "#,##0.00"})
            note_title_fmt = workbook.add_format({"bold": True, "font_color": "#03547C", "align": "left"})
            note_fmt = workbook.add_format({"font_color": "#5B6573", "text_wrap": True, "valign": "top", "align": "left"})

            for item in items:
                table = item["table"].copy()
                if table.empty:
                    continue
                sheet = _report_table_sheet_name(item["title"], used_names)
                table.to_excel(writer, sheet_name=sheet, index=False, startrow=1)
                worksheet = writer.sheets[sheet]
                worksheet.hide_gridlines(2)
                worksheet.write(0, 0, item["title"], title_fmt)
                if len(table.columns) > 1:
                    try:
                        worksheet.merge_range(0, 0, 0, len(table.columns) - 1, item["title"], title_fmt)
                    except Exception:
                        pass

                worksheet.set_row(1, 24)
                column_widths: list[int] = []
                for col_idx, column in enumerate(table.columns):
                    column_name = str(column)
                    width = _excel_column_width(table, column)
                    column_widths.append(width)
                    worksheet.write(1, col_idx, column_name, header_fmt)
                    if _is_report_numeric_series(table[column]):
                        worksheet.set_column(col_idx, col_idx, width, body_num_fmt)
                    elif _is_report_text_column(column_name):
                        worksheet.set_column(col_idx, col_idx, width, body_text_fmt)
                    else:
                        worksheet.set_column(col_idx, col_idx, width, body_fmt)

                notes = item.get("notes") or []
                if notes:
                    note_start = len(table) + 4
                    worksheet.write(note_start, 0, "Footnotes / filters", note_title_fmt)
                    usable_chars = max(45, int(sum(column_widths) * 1.15))
                    for offset, note in enumerate(notes, start=1):
                        text_value = f"{offset}. {note}"
                        row_number = note_start + offset
                        estimated_lines = max(1, (len(text_value) // usable_chars) + 1)
                        worksheet.set_row(row_number, min(72, 15 * estimated_lines))
                        if len(table.columns) > 1:
                            try:
                                worksheet.merge_range(row_number, 0, row_number, len(table.columns) - 1, text_value, note_fmt)
                            except Exception:
                                worksheet.write(row_number, 0, text_value, note_fmt)
                        else:
                            worksheet.write(row_number, 0, text_value, note_fmt)
                worksheet.freeze_panes(2, 0)
    except Exception:
        sheets = {item["title"][:31]: item["table"] for item in items if not item["table"].empty}
        return dataframe_to_excel_bytes(sheets)
    output.seek(0)
    return output.getvalue()

def _pdf_cell(value: Any, max_chars: int = 48) -> str:
    try:
        missing = value is None or bool(pd.isna(value))
    except Exception:
        missing = value is None
    if missing:
        return "-"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if abs(float(value)) >= 1000:
            text_value = f"{float(value):,.0f}"
        else:
            text_value = f"{float(value):,.3f}".rstrip("0").rstrip(".")
    else:
        text_value = str(value)
    text_value = text_value.replace("\n", " ").strip()
    return text_value if len(text_value) <= max_chars else text_value[: max_chars - 3] + "..."


def _pdf_table_data(
    table: pd.DataFrame,
    header_style: Any,
    cell_style: Any,
    numeric_cell_style: Any,
) -> list[list[Any]]:
    """Return ReportLab-ready table data using paragraphs for wrapped cells."""
    from reportlab.platypus import Paragraph

    if table is None or table.empty:
        return []

    data: list[list[Any]] = []
    data.append([Paragraph(html.escape(str(column)), header_style) for column in table.columns])

    numeric_columns = {
        column for column in table.columns
        if column in table.columns and _is_report_numeric_series(table[column])
    }

    for _, row in table.iterrows():
        row_cells: list[Any] = []
        for column in table.columns:
            is_text_column = _is_report_text_column(column)
            max_chars = 115 if is_text_column else 58
            value = _pdf_cell(row.get(column), max_chars=max_chars)
            style = numeric_cell_style if column in numeric_columns else cell_style
            row_cells.append(Paragraph(html.escape(value), style))
        data.append(row_cells)
    return data


def _pdf_column_widths(table: pd.DataFrame, available_width: float) -> list[float]:
    """Return compact, content-driven widths and only shrink when the table exceeds the page."""
    from reportlab.lib.units import inch

    if table is None or table.empty or len(table.columns) == 0:
        return []

    widths: list[float] = []
    for column in table.columns:
        name = str(column)
        values = table[column].head(500).tolist() if column in table.columns else []
        lengths = [len(name)] + [len(str(value).replace("\n", " ")) for value in values if value is not None and not pd.isna(value)]
        lengths = sorted(lengths)
        representative = lengths[min(len(lengths) - 1, int(round((len(lengths) - 1) * 0.90)))] if lengths else len(name)

        if _is_report_text_column(name):
            width = min(max(0.78 * inch, 16 + representative * 3.15), 2.35 * inch)
        elif column in table.columns and _is_report_numeric_series(table[column]):
            width = min(max(0.52 * inch, 12 + representative * 3.05), 1.10 * inch)
        else:
            width = min(max(0.62 * inch, 14 + representative * 3.10), 1.55 * inch)
        widths.append(float(width))

    total = sum(widths)
    if total > available_width and total > 0:
        scale = available_width / total
        widths = [width * scale for width in widths]
    return widths

def _report_chart_color(color: str) -> str:
    """Darken very light neutral colors so report charts retain contrast in PDF output."""
    value = str(color or "#A5A5A5").strip()
    if not re.fullmatch(r"#[0-9A-Fa-f]{6}", value):
        return value
    red = int(value[1:3], 16)
    green = int(value[3:5], 16)
    blue = int(value[5:7], 16)
    average = (red + green + blue) / 3.0
    is_neutral = max(red, green, blue) - min(red, green, blue) <= 24
    if is_neutral and 155 <= average < 245:
        factor = 0.68
        red = max(0, min(255, int(red * factor)))
        green = max(0, min(255, int(green * factor)))
        blue = max(0, min(255, int(blue * factor)))
        return f"#{red:02X}{green:02X}{blue:02X}"
    return value


def _report_chart_colors(colors: list[str] | None, count: int) -> list[str] | None:
    if not colors:
        return colors
    normalized = [_report_chart_color(color) for color in colors]
    if not normalized:
        return normalized
    return [normalized[index % len(normalized)] for index in range(count)]


def _style_report_axis(ax, x_label: str = "", y_label: str = "", rotation: int = 45) -> None:
    """Apply a consistent, high-contrast style to charts exported to PDF."""
    dark = "#26323A"
    grid = "#AAB4BE"
    # PNGs are scaled down to fit an A4 page. These source sizes produce
    # approximately 10–11 pt text in the final PDF instead of 7–8 pt.
    ax.set_xlabel(x_label, fontsize=18, fontweight="semibold", color=dark, labelpad=7)
    ax.set_ylabel(y_label, fontsize=18, fontweight="semibold", color=dark, labelpad=8)
    # Keep tick labels close to the plotting area and large enough for PDF review.
    # A small negative pad plus anchored rotation avoids the large visual gap that
    # Matplotlib otherwise leaves below categorical/date axes in exported PNGs.
    ax.tick_params(axis="x", labelrotation=rotation, labelsize=11, colors=dark, pad=-3)
    ax.tick_params(axis="y", labelsize=11, colors=dark, pad=2)
    for label in ax.get_xticklabels():
        label.set_rotation_mode("anchor")
        label.set_horizontalalignment("right" if rotation else "center")
        label.set_verticalalignment("top")
    ax.grid(axis="y", color=grid, alpha=0.60, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#6F7B86")
        ax.spines[spine].set_linewidth(0.9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def _place_report_legend(ax, handles=None, labels=None, ncol: int = 4):
    """Place a readable legend inside the plot area with a white framed background."""
    kwargs = {
        "loc": "upper center",
        "bbox_to_anchor": (0.5, 0.985),
        "ncol": max(1, ncol),
        "fontsize": 11,
        "frameon": True,
        "framealpha": 0.85,
        "facecolor": "white",
        "edgecolor": "#7F8B96",
        "borderpad": 0.55,
        "columnspacing": 1.15,
        "handlelength": 1.7,
        "handletextpad": 0.45,
    }
    legend = ax.legend(handles=handles, labels=labels, **kwargs) if handles is not None else ax.legend(**kwargs)
    if legend is not None:
        for value in legend.get_texts():
            value.set_color("#26323A")
    return legend


def _pillow_font(size: int, bold: bool = False):
    """Return a portable Pillow font without requiring project font files."""
    from PIL import ImageFont

    candidates = [
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        "Arial Bold.ttf" if bold else "Arial.ttf",
        "arialbd.ttf" if bold else "arial.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def _pillow_png_from_pivot(
    pivot: pd.DataFrame,
    title: str,
    y_label: str,
    colors: list[str] | None = None,
    stacked: bool = True,
    line_values: list[float] | None = None,
    line_label: str | None = None,
    line_color: str = "#A39161",
) -> bytes | None:
    """Render a compact bar chart with Pillow when Matplotlib is unavailable."""
    if pivot is None or pivot.empty:
        return None
    try:
        from PIL import Image, ImageDraw
    except Exception:
        return None

    numeric = pivot.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    labels = [str(value) for value in numeric.index.tolist()]
    series_names = [str(value) for value in numeric.columns.tolist()]
    if not labels or not series_names:
        return None

    palette = colors or ["#03547C", "#A39161", "#70AD47", "#70767C", "#ED7D31", "#5B9BD5"]
    palette = [_report_chart_color(palette[index % len(palette)]) for index in range(len(series_names))]

    clean_line = (
        [float(value) if value is not None and pd.notna(value) else 0.0 for value in line_values]
        if line_values and len(line_values) == len(labels)
        else []
    )
    line_max = max(clean_line) if clean_line else 0.0
    has_secondary_axis = line_max > 0 and bool(line_label)
    secondary_axis_color = "#C08A00"
    secondary_grid_color = "#D6B45A"

    chart_width = max(1450, min(2700, 105 * len(labels) + 420))
    chart_height = 760
    image = Image.new("RGB", (chart_width, chart_height), "white")
    draw = ImageDraw.Draw(image)

    title_font = _pillow_font(34, bold=True)
    axis_font = _pillow_font(23, bold=True)
    tick_font = _pillow_font(19, bold=False)
    legend_font = _pillow_font(18, bold=False)

    left, right, top, bottom = 125, (145 if has_secondary_axis else 55), 85, 155
    plot_left, plot_right = left, chart_width - right
    plot_top, plot_bottom = top, chart_height - bottom
    plot_width = max(1, plot_right - plot_left)
    plot_height = max(1, plot_bottom - plot_top)

    title_bbox = draw.textbbox((0, 0), title, font=title_font)
    draw.text(((chart_width - (title_bbox[2] - title_bbox[0])) / 2, 22), title, fill="#202A36", font=title_font)

    if stacked:
        max_value = float(numeric.sum(axis=1).max())
    else:
        max_value = float(numeric.max(axis=1).max())
    max_value = max(max_value, 1.0)
    y_max = max_value * 1.30

    # Grid and y-axis labels.
    for step in range(6):
        value = y_max * step / 5
        y = plot_bottom - (value / y_max) * plot_height
        draw.line((plot_left, y, plot_right, y), fill="#AEB8C2", width=2)
        if value >= 1_000_000:
            label = f"{value / 1_000_000:.1f}M"
        elif value >= 1_000:
            label = f"{value / 1_000:.0f}K"
        else:
            label = f"{value:,.0f}"
        bbox = draw.textbbox((0, 0), label, font=tick_font)
        draw.text((plot_left - 12 - (bbox[2] - bbox[0]), y - 11), label, fill="#26323A", font=tick_font)

    if has_secondary_axis:
        secondary_max = line_max * 1.10
        for step in range(6):
            value = secondary_max * step / 5
            y = plot_bottom - (value / secondary_max) * plot_height
            # Dotted/dashed grid line for the secondary Au-grade axis.
            dash_x = plot_left
            while dash_x < plot_right:
                dash_end = min(plot_right, dash_x + 9)
                draw.line((dash_x, y, dash_end, y), fill=secondary_grid_color, width=2)
                dash_x += 18
            label = f"{value:.2f}" if secondary_max < 10 else f"{value:.1f}"
            draw.text((plot_right + 12, y - 11), label, fill=secondary_axis_color, font=tick_font)
        draw.line((plot_right, plot_top, plot_right, plot_bottom), fill=secondary_axis_color, width=2)

    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill="#82909F", width=2)
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill="#82909F", width=2)

    n_labels = len(labels)
    group_width = plot_width / max(1, n_labels)
    outer_gap = max(2.0, group_width * 0.15)

    for index, label in enumerate(labels):
        x0 = plot_left + index * group_width + outer_gap
        x1 = plot_left + (index + 1) * group_width - outer_gap
        if stacked:
            cumulative = 0.0
            for series_idx, series_name in enumerate(series_names):
                value = max(0.0, float(numeric.iloc[index, series_idx]))
                y1 = plot_bottom - (cumulative / y_max) * plot_height
                cumulative += value
                y0 = plot_bottom - (cumulative / y_max) * plot_height
                if y1 > y0:
                    draw.rectangle((x0, y0, x1, y1), fill=palette[series_idx], outline="white", width=1)
        else:
            bar_gap = max(1.0, (x1 - x0) * 0.04)
            bar_width = max(2.0, ((x1 - x0) - bar_gap * (len(series_names) - 1)) / max(1, len(series_names)))
            for series_idx, series_name in enumerate(series_names):
                value = max(0.0, float(numeric.iloc[index, series_idx]))
                bx0 = x0 + series_idx * (bar_width + bar_gap)
                bx1 = bx0 + bar_width
                by0 = plot_bottom - (value / y_max) * plot_height
                draw.rectangle((bx0, by0, bx1, plot_bottom), fill=palette[series_idx], outline="white", width=1)

        # Keep labels legible by thinning when there are many categories.
        show_every = max(1, (n_labels + 22) // 23)
        if index % show_every == 0 or index == n_labels - 1:
            label_img = Image.new("RGBA", (180, 48), (255, 255, 255, 0))
            label_draw = ImageDraw.Draw(label_img)
            label_draw.text((2, 2), label[:18], fill="#26323A", font=tick_font)
            rotated = label_img.rotate(35, expand=True, resample=Image.Resampling.BICUBIC)
            image.paste(rotated, (int((x0 + x1) / 2 - rotated.width / 2), int(plot_bottom + 2)), rotated)

    # Optional grade line on a secondary scale.
    if has_secondary_axis:
        secondary_max = line_max * 1.10
        points = []
        for index, value in enumerate(clean_line):
            x = plot_left + (index + 0.5) * group_width
            y = plot_bottom - (value / secondary_max) * plot_height
            points.append((x, y))
        if len(points) > 1:
            draw.line(points, fill=secondary_axis_color, width=5, joint="curve")
        for x, y in points:
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=secondary_axis_color, outline="white", width=1)

    # Axis title.
    y_bbox = draw.textbbox((0, 0), y_label, font=axis_font)
    y_img = Image.new("RGBA", (y_bbox[2] - y_bbox[0] + 16, y_bbox[3] - y_bbox[1] + 16), (255, 255, 255, 0))
    y_draw = ImageDraw.Draw(y_img)
    y_draw.text((8, 8), y_label, fill="#26323A", font=axis_font)
    y_rotated = y_img.rotate(90, expand=True)
    image.paste(y_rotated, (15, int(plot_top + plot_height / 2 - y_rotated.height / 2)), y_rotated)

    if has_secondary_axis and line_label:
        secondary_bbox = draw.textbbox((0, 0), line_label, font=axis_font)
        secondary_img = Image.new(
            "RGBA",
            (secondary_bbox[2] - secondary_bbox[0] + 16, secondary_bbox[3] - secondary_bbox[1] + 16),
            (255, 255, 255, 0),
        )
        secondary_draw = ImageDraw.Draw(secondary_img)
        secondary_draw.text((8, 8), line_label, fill=secondary_axis_color, font=axis_font)
        secondary_rotated = secondary_img.rotate(270, expand=True)
        image.paste(
            secondary_rotated,
            (chart_width - secondary_rotated.width - 10, int(plot_top + plot_height / 2 - secondary_rotated.height / 2)),
            secondary_rotated,
        )

    # Legend.
    legend_entries = list(zip(series_names, palette))
    if line_label:
        legend_entries.append((line_label, line_color))
    total_legend_width = 0
    entry_widths = []
    for name, _ in legend_entries:
        bbox = draw.textbbox((0, 0), name, font=legend_font)
        width = 28 + (bbox[2] - bbox[0]) + 24
        entry_widths.append(width)
        total_legend_width += width
    legend_x = max(20, (chart_width - total_legend_width) / 2)
    legend_y = plot_top + 12
    for (name, color), entry_width in zip(legend_entries, entry_widths):
        draw.rectangle((legend_x, legend_y, legend_x + 16, legend_y + 16), fill=color)
        draw.text((legend_x + 23, legend_y - 3), name, fill="#26323A", font=legend_font)
        legend_x += entry_width

    output = io.BytesIO()
    image.save(output, format="PNG", optimize=True)
    output.seek(0)
    return output.getvalue()


def _pillow_pie_chart_png(
    labels: list[str],
    values: list[float],
    colors: list[str],
    title: str,
) -> bytes | None:
    """Render a donut chart with Pillow when Matplotlib is unavailable."""
    if not labels or not values or sum(max(0.0, float(v)) for v in values) <= 0:
        return None
    try:
        from PIL import Image, ImageDraw
    except Exception:
        return None

    colors = [_report_chart_color(color) for color in colors]
    width, height = 1250, 700
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = _pillow_font(30, bold=True)
    label_font = _pillow_font(18, bold=False)
    value_font = _pillow_font(17, bold=True)

    title_bbox = draw.textbbox((0, 0), title, font=title_font)
    draw.text(((width - (title_bbox[2] - title_bbox[0])) / 2, 25), title, fill="#202A36", font=title_font)

    box = (110, 115, 650, 655)
    total = float(sum(max(0.0, float(v)) for v in values))
    start_angle = -90.0
    for label, value, color in zip(labels, values, colors):
        fraction = max(0.0, float(value)) / total
        end_angle = start_angle + fraction * 360.0
        draw.pieslice(box, start=start_angle, end=end_angle, fill=color, outline="white", width=2)
        start_angle = end_angle
    draw.ellipse((250, 255, 510, 515), fill="white")
    total_text = f"{total:,.1f}"
    bbox = draw.textbbox((0, 0), total_text, font=value_font)
    draw.text((380 - (bbox[2] - bbox[0]) / 2, 365), total_text, fill="#202A36", font=value_font)

    legend_x, legend_y = 735, 145
    for index, (label, value, color) in enumerate(zip(labels, values, colors)):
        y = legend_y + index * 49
        pct = 100.0 * max(0.0, float(value)) / total
        draw.rectangle((legend_x, y, legend_x + 20, y + 20), fill=color)
        draw.text((legend_x + 32, y - 2), f"{label}  {pct:.0f}%", fill="#26323A", font=label_font)

    output = io.BytesIO()
    image.save(output, format="PNG", optimize=True)
    output.seek(0)
    return output.getvalue()


def _plot_png_from_pivot(pivot: pd.DataFrame, title: str, y_label: str, colors: list[str] | None = None, stacked: bool = True) -> bytes | None:
    if pivot.empty:
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return _pillow_png_from_pivot(pivot, title, y_label, colors, stacked)

    try:
        plot_colors = _report_chart_colors(colors, len(pivot.columns))
        fig, ax = plt.subplots(figsize=(12.8, 4.6))
        pivot.plot(kind="bar", stacked=stacked, ax=ax, color=plot_colors, width=0.82)
        ax.set_title(title, fontsize=23, fontweight="bold", color="#202A36", pad=14)
        _style_report_axis(ax, x_label="", y_label=y_label, rotation=45)

        current_top = float(ax.get_ylim()[1]) if ax.get_ylim()[1] else 1.0
        if "%" in y_label:
            ax.set_ylim(0, max(112.0, current_top * 1.08))
        else:
            ax.set_ylim(0, current_top * 1.18)

        _place_report_legend(ax, ncol=min(5, max(1, len(pivot.columns))))
        fig.subplots_adjust(left=0.075, right=0.985, top=0.88, bottom=0.145)
        output = io.BytesIO()
        fig.savefig(output, format="png", dpi=195, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        output.seek(0)
        return output.getvalue()
    except Exception:
        try:
            plt.close("all")
        except Exception:
            pass
        return _pillow_png_from_pivot(pivot, title, y_label, colors, stacked)


def _resource_category_bench_chart_png(data: pd.DataFrame, config: ModelConfig, model_name: str, destination_mode: str) -> bytes | None:
    bench_col = config.column_for_role("Bench")
    category_col = config.column_for_role("Category")
    if not bench_col or not category_col or bench_col not in data.columns or category_col not in data.columns or data.empty:
        return None
    table = _group_tonnes(data, config, [bench_col, category_col])
    if table.empty:
        return None
    ton_col = tonnage_column_name(config)
    table["Bench"] = table[bench_col].astype(str)
    table["Category"] = table[category_col].map(_display_resource_category)
    pivot = table.pivot_table(index="Bench", columns="Category", values=ton_col, aggfunc="sum", fill_value=0)
    try:
        order = sorted(pivot.index.tolist(), key=lambda x: float(x), reverse=True)
        pivot = pivot.loc[order]
    except Exception:
        pivot = pivot.sort_index(ascending=False)
    pivot = pivot[[col for col in RESOURCE_CATEGORY_ORDER if col in pivot.columns] + [col for col in pivot.columns if col not in RESOURCE_CATEGORY_ORDER]]
    row_sum = pivot.sum(axis=1).replace(0, pd.NA)
    pivot_pct = pivot.div(row_sum, axis=0).fillna(0) * 100.0
    colors = [RESOURCE_CATEGORY_COLORS.get(col, "#A5A5A5") for col in pivot_pct.columns]
    return _plot_png_from_pivot(pivot_pct, f"Resource Category Distribution by Bench - {model_name} ({destination_mode})", "Distribution (%)", colors, stacked=True)


def _metal_at_risk_chart_png(data: pd.DataFrame, config: ModelConfig, model_name: str) -> bytes | None:
    bench_col = config.column_for_role("Bench")
    category_col = config.column_for_role("Category")
    au_spec = _spec_by_canonical_label(config, "Au")
    if not bench_col or not category_col or not au_spec or bench_col not in data.columns or category_col not in data.columns or au_spec.column not in data.columns:
        return None
    rows: list[dict[str, Any]] = []
    for category_key, display_label in [("grade_control", "Grade Control"), ("measured", "Measured"), ("indicated", "Indicated (At Risk)")]:
        subset = data[_category_mask(data, category_col, category_key)]
        if subset.empty:
            continue
        for bench, group in subset.groupby(bench_col, dropna=False, observed=True):
            metal = contained_metal(group[au_spec.column], group[config.mass_column], _effective_grade_unit(au_spec))
            ounces = float(metal[0]) if metal and metal[1] == "oz" else 0.0
            rows.append({"Bench": str(bench), "Category": display_label, "Au oz": ounces})
    if not rows:
        return None
    table = pd.DataFrame(rows)
    max_oz = float(table.groupby("Bench")["Au oz"].sum().max())
    unit, divisor, _ = _metal_at_risk_display_unit(max_oz)
    table[f"Au ({unit})"] = table["Au oz"] / divisor
    pivot = table.pivot_table(index="Bench", columns="Category", values=f"Au ({unit})", aggfunc="sum", fill_value=0)
    try:
        order = sorted(pivot.index.tolist(), key=lambda x: float(x), reverse=True)
        pivot = pivot.loc[order]
    except Exception:
        pivot = pivot.sort_index(ascending=False)
    colors = ["#485B73", "#A39161", "#BFC3C8"]
    return _plot_png_from_pivot(pivot, f"Metal at Risk - {_active_year_scope_short_label()} - {model_name}", f"Au ({unit})", colors, stacked=True)


def _comparison_ore_phase_chart_png(table: pd.DataFrame) -> bytes | None:
    if table.empty:
        return None
    pivot = table.pivot_table(index="Phase", columns="Model", values="Ore Kt", aggfunc="sum", fill_value=0)
    try:
        order = sorted(pivot.index.tolist(), key=_natural_phase_sort_key)
        pivot = pivot.loc[order]
    except Exception:
        pivot = pivot.sort_index()
    colors = ["#03547C", "#A39161", *MODEL_COLORS][: len(pivot.columns)]
    return _plot_png_from_pivot(pivot, "Ore Kt by Phase", "Ore Kt", colors, stacked=False)



def _role_bench_chart_png(data: pd.DataFrame, config: ModelConfig, role: str, model_name: str, destination_mode: str, color_map: dict[str, str], percent: bool = True) -> bytes | None:
    bench_col = config.column_for_role("Bench")
    role_col = config.column_for_role(role)
    if not bench_col or not role_col or bench_col not in data.columns or role_col not in data.columns or data.empty:
        return None
    table = _group_tonnes(data, config, [bench_col, role_col])
    if table.empty:
        return None
    ton_col = tonnage_column_name(config)
    table["Bench"] = table[bench_col].astype(str)
    if role == "Category":
        table[role] = table[role_col].map(_display_resource_category)
        order_cols = RESOURCE_CATEGORY_ORDER
    elif role == "Destination":
        table[role] = table[role_col].map(_display_destination)
        order_cols = ["HG", "LG", "MG", "MW", "Waste", *sorted(table[role].dropna().astype(str).unique().tolist())]
    else:
        table[role] = table[role_col].astype(str).str.strip().str.upper()
        order_cols = sorted(table[role].dropna().astype(str).unique().tolist())
    pivot = table.pivot_table(index="Bench", columns=role, values=ton_col, aggfunc="sum", fill_value=0)
    try:
        order = sorted(pivot.index.tolist(), key=lambda x: float(x), reverse=True)
        pivot = pivot.loc[order]
    except Exception:
        pivot = pivot.sort_index(ascending=False)
    ordered = [col for col in order_cols if col in pivot.columns]
    pivot = pivot[ordered + [col for col in pivot.columns if col not in ordered]]
    if percent:
        row_sum = pivot.sum(axis=1).replace(0, pd.NA)
        pivot = pivot.div(row_sum, axis=0).fillna(0) * 100.0
        y_label = "Distribution (%)"
    else:
        y_label = ton_col
    colors = [color_map.get(str(col), "#A5A5A5") for col in pivot.columns]
    return _plot_png_from_pivot(pivot, f"{role} Distribution by Bench - {model_name} ({destination_mode})", y_label, colors, stacked=True)


def _role_pie_chart_png(data: pd.DataFrame, config: ModelConfig, role: str, model_name: str, title: str, color_map: dict[str, str]) -> bytes | None:
    role_col = config.column_for_role(role)
    if not role_col or role_col not in data.columns or data.empty:
        return None
    table = _group_tonnes(data, config, [role_col])
    if table.empty:
        return None
    ton_col = tonnage_column_name(config)
    if role == "Category":
        table[role] = table[role_col].map(_display_resource_category)
        order = RESOURCE_CATEGORY_ORDER
    elif role == "Destination":
        table[role] = table[role_col].map(_display_destination)
        order = ["HG", "LG", "MG", "MW", "Waste"]
    else:
        table[role] = table[role_col].astype(str).str.strip().str.upper()
        order = sorted(table[role].dropna().unique().tolist())
    table = table.groupby(role, dropna=False, observed=True)[ton_col].sum().reset_index()
    table["__order__"] = table[role].map({name: idx for idx, name in enumerate(order)}).fillna(len(order))
    table = table.sort_values("__order__").drop(columns="__order__")
    if table[ton_col].sum() <= 0:
        return None
    labels = table[role].astype(str).tolist()
    values = [float(value) for value in table[ton_col].tolist()]
    colors = [color_map.get(label, "#A5A5A5") for label in labels]
    chart_title = f"{title} - {model_name}"
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return _pillow_pie_chart_png(labels, values, colors, chart_title)
    try:
        plot_colors = [_report_chart_color(color) for color in colors]
        fig, ax = plt.subplots(figsize=(8.8, 4.8))
        wedges, texts, autotexts = ax.pie(
            values,
            labels=labels,
            colors=plot_colors,
            autopct=lambda pct: f"{pct:.0f}%" if pct >= 2 else "",
            startangle=90,
            pctdistance=0.75,
            labeldistance=1.08,
            textprops={"fontsize": 13.5, "color": "#26323A"},
            wedgeprops={"linewidth": 0.8, "edgecolor": "white"},
        )
        for value in autotexts:
            value.set_fontsize(13.5)
            value.set_fontweight("bold")
            value.set_color("#202A36")
        centre = plt.Circle((0, 0), 0.55, fc="white")
        ax.add_artist(centre)
        ax.set_title(chart_title, fontsize=21, fontweight="bold", color="#202A36", pad=14)
        ax.axis("equal")
        fig.tight_layout()
        output = io.BytesIO()
        fig.savefig(output, format="png", dpi=195, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        output.seek(0)
        return output.getvalue()
    except Exception:
        try:
            plt.close("all")
        except Exception:
            pass
        return _pillow_pie_chart_png(labels, values, colors, chart_title)


def _ore_phase_chart_png(data: pd.DataFrame, config: ModelConfig, model_name: str) -> bytes | None:
    phase_col = _phase_column(config, data)
    dest_col = config.column_for_role("Destination")
    if not phase_col or not dest_col or phase_col not in data.columns or dest_col not in data.columns or data.empty:
        return None
    work = data.copy()
    work["__dest_code__"] = work[dest_col].map(_normalize_destination_code)
    ore = work[work["__dest_code__"].isin(ORE_DESTINATION_CODES)].copy()
    if ore.empty:
        return None
    ore[config.mass_column] = pd.to_numeric(ore[config.mass_column], errors="coerce").fillna(0).clip(lower=0)
    table = ore.groupby(phase_col, dropna=False, observed=True)[config.mass_column].sum().reset_index()
    table["Phase"] = table[phase_col].astype(str).str.strip()
    table["Ore Kt"] = table[config.mass_column] / 1_000.0
    pivot = table.pivot_table(index="Phase", values="Ore Kt", aggfunc="sum", fill_value=0)
    try:
        pivot = pivot.loc[sorted(pivot.index.tolist(), key=_natural_phase_sort_key)]
    except Exception:
        pivot = pivot.sort_index()
    return _plot_png_from_pivot(pivot, f"Ore Kt by Phase - {model_name}", "Ore Kt", ["#03547C"], stacked=False)


def _year_role_chart_png(data: pd.DataFrame, config: ModelConfig, role: str, model_name: str, destination_mode: str, color_map: dict[str, str], percent: bool = True) -> bytes | None:
    year_col = _year_column(config, data)
    role_col = config.column_for_role(role)
    if not year_col or not role_col or year_col not in data.columns or role_col not in data.columns or data.empty:
        return None
    table = _group_tonnes(data, config, [year_col, role_col])
    if table.empty:
        return None
    ton_col = tonnage_column_name(config)
    table["Year"] = pd.to_numeric(table[year_col], errors="coerce")
    table = table[table["Year"].notna()].copy()
    if table.empty:
        return None
    table["Year"] = table["Year"].astype(int).astype(str)
    if role == "Category":
        table[role] = table[role_col].map(_display_resource_category)
        order_cols = RESOURCE_CATEGORY_ORDER
    elif role == "Destination":
        table[role] = table[role_col].map(_display_destination)
        order_cols = ["HG", "LG", "MG", "MW", "Waste"]
    else:
        table[role] = table[role_col].astype(str).str.strip().str.upper()
        order_cols = sorted(table[role].dropna().astype(str).unique().tolist())
    pivot = table.pivot_table(index="Year", columns=role, values=ton_col, aggfunc="sum", fill_value=0)
    try:
        pivot = pivot.loc[sorted(pivot.index.tolist(), key=lambda x: int(x))]
    except Exception:
        pivot = pivot.sort_index()
    pivot = pivot[[col for col in order_cols if col in pivot.columns] + [col for col in pivot.columns if col not in order_cols]]
    if percent:
        row_sum = pivot.sum(axis=1).replace(0, pd.NA)
        pivot = pivot.div(row_sum, axis=0).fillna(0) * 100.0
        y_label = "Distribution (%)"
    else:
        y_label = ton_col
    colors = [color_map.get(str(col), "#A5A5A5") for col in pivot.columns]
    return _plot_png_from_pivot(pivot, f"{role} Distribution by Year - {model_name} ({destination_mode})", y_label, colors, stacked=True)


def _destination_bench_grade_chart_png(data: pd.DataFrame, config: ModelConfig, model_name: str, destination_mode: str) -> bytes | None:
    bench_col = config.column_for_role("Bench")
    dest_col = config.column_for_role("Destination")
    if not bench_col or not dest_col or bench_col not in data.columns or dest_col not in data.columns or data.empty:
        return None
    table = _group_tonnes(data, config, [bench_col, dest_col])
    if table.empty:
        return None
    ton_col = tonnage_column_name(config)
    table["Bench"] = table[bench_col].astype(str)
    table["Destination"] = table[dest_col].map(_display_destination)
    pivot = table.pivot_table(index="Bench", columns="Destination", values=ton_col, aggfunc="sum", fill_value=0)
    try:
        pivot = pivot.loc[sorted(pivot.index.tolist(), key=lambda x: float(x), reverse=True)]
    except Exception:
        pivot = pivot.sort_index(ascending=False)
    ordered_cols = [col for col in ["HG", "LG", "MG", "MW", "Waste"] if col in pivot.columns]
    pivot = pivot[ordered_cols + [col for col in pivot.columns if col not in ordered_cols]]
    colors = [DESTINATION_COLORS.get(str(col), "#A5A5A5") for col in pivot.columns]
    grade_values: list[float] | None = None
    grade_label: str | None = None
    au_spec = next((spec for spec in config.grade_specs if spec.label.casefold() == "au" and not _is_contained_metal_spec(spec)), None)
    if au_spec and au_spec.column in data.columns:
        grade_rows = []
        for bench, group in data.groupby(bench_col, observed=True):
            grade_rows.append({"Bench": str(bench), f"Au ({au_spec.unit})": weighted_mean(group[au_spec.column], group[config.mass_column])})
        grade_table = pd.DataFrame(grade_rows)
        if not grade_table.empty:
            grade_table = grade_table.set_index("Bench").reindex(pivot.index)
            grade_values = pd.to_numeric(grade_table.iloc[:, 0], errors="coerce").fillna(0).tolist()
            grade_label = f"Au ({au_spec.unit})"
    chart_title = f"Destination Ore Distribution by Bench - {model_name} ({destination_mode})"
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return _pillow_png_from_pivot(
            pivot,
            chart_title,
            ton_col,
            colors,
            stacked=True,
            line_values=grade_values,
            line_label=grade_label,
            line_color="#C08A00",
        )
    try:
        plot_colors = _report_chart_colors(colors, len(pivot.columns))
        fig, ax = plt.subplots(figsize=(12.8, 4.6))
        pivot.plot(kind="bar", stacked=True, ax=ax, color=plot_colors, width=0.82)
        ax.set_title(chart_title, fontsize=15, fontweight="bold", color="#202A36", pad=12)
        _style_report_axis(ax, x_label="Bench", y_label=ton_col, rotation=45)
        ax.set_ylim(0, float(ax.get_ylim()[1]) * 1.20)

        combined_handles, combined_labels = ax.get_legend_handles_labels()
        if grade_values is not None and grade_label:
            secondary_axis_color = "#C08A00"
            ax2 = ax.twinx()
            ax2.plot(
                range(len(pivot.index)),
                grade_values,
                color=secondary_axis_color,
                marker="o",
                markerfacecolor=secondary_axis_color,
                markeredgecolor="white",
                markeredgewidth=0.7,
                markersize=4.8,
                linewidth=2.2,
                label=grade_label,
                zorder=5,
            )
            ax2.set_ylabel(
                grade_label,
                fontsize=18,
                fontweight="semibold",
                color=secondary_axis_color,
                labelpad=7,
            )
            ax2.tick_params(
                axis="y",
                labelsize=16.5,
                colors=secondary_axis_color,
                color=secondary_axis_color,
                pad=2,
            )
            ax2.spines["right"].set_color(secondary_axis_color)
            ax2.spines["right"].set_linewidth(1.1)
            ax2.spines["right"].set_linestyle((0, (4, 3)))
            ax2.spines["top"].set_visible(False)
            ax2.patch.set_visible(False)
            ax2.set_axisbelow(True)
            # Secondary-axis grid: dotted gold lines, distinct from the solid
            # gray grid of the primary tonnage axis.
            ax2.grid(
                axis="y",
                color=secondary_axis_color,
                linestyle=(0, (2.2, 3.2)),
                alpha=0.34,
                linewidth=0.9,
            )
            second_handles, second_labels = ax2.get_legend_handles_labels()
            combined_handles += second_handles
            combined_labels += second_labels

        _place_report_legend(
            ax,
            handles=combined_handles,
            labels=combined_labels,
            ncol=min(5, max(1, len(combined_labels))),
        )
        fig.subplots_adjust(left=0.075, right=0.92, top=0.88, bottom=0.145)
        output = io.BytesIO()
        fig.savefig(output, format="png", dpi=195, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        output.seek(0)
        return output.getvalue()
    except Exception:
        try:
            plt.close("all")
        except Exception:
            pass
        return _pillow_png_from_pivot(
            pivot,
            chart_title,
            ton_col,
            colors,
            stacked=True,
            line_values=grade_values,
            line_label=grade_label,
        )


def _grade_distribution_multiplot_png(data: pd.DataFrame, config: ModelConfig, model_name: str) -> bytes | None:
    specs = [spec for spec in config.grade_specs if spec.column in data.columns and not _is_contained_metal_spec(spec)][:9]
    if not specs or data.empty:
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    n = len(specs)
    cols = 3 if n > 2 else n
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12.8, max(3.4, 3.0 * rows)))
    if not isinstance(axes, (list, tuple)) and not hasattr(axes, "flat"):
        axes_list = [axes]
    else:
        axes_list = list(axes.flat)
    for ax, spec in zip(axes_list, specs):
        series = pd.to_numeric(data[spec.column], errors="coerce")
        series = series[series > 0]
        if series.empty:
            ax.text(0.5, 0.5, "No positive values", ha="center", va="center", fontsize=15, color="#26323A")
        else:
            ax.hist(series, bins=35, color="#03547C", edgecolor="white", linewidth=0.45)
        ax.set_title(f"{spec.label} ({spec.unit})", fontsize=16, fontweight="bold", color="#202A36", pad=9)
        ax.tick_params(axis="both", labelsize=15, colors="#26323A", pad=2)
        ax.grid(axis="y", color="#AAB4BE", alpha=0.55, linewidth=0.75)
        ax.set_axisbelow(True)
        ax.spines["left"].set_color("#6F7B86")
        ax.spines["bottom"].set_color("#6F7B86")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    for ax in axes_list[len(specs):]:
        ax.axis("off")
    fig.suptitle(f"Grade Distribution - {model_name}", fontsize=22, fontweight="bold", color="#202A36")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output = io.BytesIO()
    fig.savefig(output, format="png", dpi=195, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    output.seek(0)
    return output.getvalue()


def _automatic_report_charts(scoped_bundles: dict[str, ModelBundle]) -> list[dict[str, Any]]:
    charts: list[dict[str, Any]] = []
    for model_name, bundle in scoped_bundles.items():
        destination_mode = _report_destination_mode_for_model(model_name)
        resource_data = _apply_destination_mode(bundle.data, bundle.config, destination_mode)
        raw_bundle = st.session_state.models.get(model_name, bundle)

        def chart_notes(extra_note: str) -> list[str]:
            notes = _table_filter_note_items(
                bundle.config,
                source_data=raw_bundle.data,
                final_data=resource_data,
                model_name=model_name,
                extra_items=[extra_note],
            )
            return [note.replace("Rows used in table:", "Rows used in plot:") for note in notes]

        chart_specs: list[tuple[str, bytes | None, list[str]]] = [
            (
                f"Met-Type Ore Distribution by Bench - {model_name}",
                _role_bench_chart_png(resource_data, bundle.config, "Mettype", model_name, destination_mode, METTYPE_COLORS, percent=True),
                chart_notes("Plot grouping: Mettype by Bench; distribution shown as percentage."),
            ),
            (
                f"Resource Category Distribution by Bench - {model_name}",
                _resource_category_bench_chart_png(resource_data, bundle.config, model_name, destination_mode),
                chart_notes("Plot grouping: Resource Category by Bench; distribution shown as percentage."),
            ),
            (
                f"Metal at Risk - {model_name}",
                _metal_at_risk_chart_png(resource_data, bundle.config, model_name),
                chart_notes("Grade Control, Measured and Indicated Au ounces by Bench."),
            ),
            (
                f"Destination Ore Distribution by Bench - {model_name}",
                _destination_bench_grade_chart_png(resource_data, bundle.config, model_name, destination_mode),
                chart_notes("Destination tonnes by Bench with Au grade on the secondary vertical axis when Au is configured."),
            ),
            (
                f"Met-Type Ore Tonnes Distribution - {model_name}",
                _role_pie_chart_png(resource_data, bundle.config, "Mettype", model_name, "Met-Type Ore Tonnes Distribution", METTYPE_COLORS),
                chart_notes("Plot grouping: ore-tonnage share by Mettype."),
            ),
            (
                f"Resource Category Tonnes Distribution - {model_name}",
                _role_pie_chart_png(resource_data, bundle.config, "Category", model_name, "Resource Category Tonnes Distribution", RESOURCE_CATEGORY_COLORS),
                chart_notes("Plot grouping: ore-tonnage share by Resource Category."),
            ),
            (
                f"Destination Tonnes Distribution - {model_name}",
                _role_pie_chart_png(resource_data, bundle.config, "Destination", model_name, "Destination Tonnes Distribution", DESTINATION_COLORS),
                chart_notes("Plot grouping: tonnage share by Destination."),
            ),
            (
                f"Ore Kt by Phase - {model_name}",
                _ore_phase_chart_png(bundle.data, bundle.config, model_name),
                chart_notes("Ore destinations only, grouped by the unified Phase/Pit_Phase field."),
            ),
            (
                f"Mettype by Year - {model_name}",
                _year_role_chart_png(resource_data, bundle.config, "Mettype", model_name, destination_mode, METTYPE_COLORS, percent=True),
                chart_notes("Annual distribution grouped by Mettype after all master filters."),
            ),
            (
                f"Category by Year - {model_name}",
                _year_role_chart_png(resource_data, bundle.config, "Category", model_name, destination_mode, RESOURCE_CATEGORY_COLORS, percent=True),
                chart_notes("Annual distribution grouped by Resource Category after all master filters."),
            ),
            (
                f"Destination by Year - {model_name}",
                _year_role_chart_png(resource_data, bundle.config, "Destination", model_name, destination_mode, DESTINATION_COLORS, percent=True),
                chart_notes("Annual distribution grouped by Destination after all master filters."),
            ),
        ]
        for title, image, notes in chart_specs:
            if image:
                charts.append({"title": title, "image": image, "notes": notes})

    if len(scoped_bundles) >= 2:
        names = list(scoped_bundles.keys())
        table = _comparison_ore_kt_by_phase_table(scoped_bundles, names)
        image = _comparison_ore_phase_chart_png(table)
        if image:
            charts.append(
                {
                    "title": "Comparison - Ore Kt by Phase",
                    "image": image,
                    "notes": _comparison_filter_note_items(
                        scoped_bundles,
                        names,
                        extra_items=["Ore destinations only; grouped by the unified Phase/Pit_Phase field and selected models."],
                    ),
                }
            )
    return charts

def _report_pdf_bytes(
    items: list[dict[str, Any]],
    charts: list[dict[str, Any]],
    model_names: list[str],
    report_name: str | None = None,
) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    from reportlab.lib.utils import ImageReader

    output = io.BytesIO()
    page_size = A4
    doc = SimpleDocTemplate(
        output,
        pagesize=page_size,
        rightMargin=0.42 * inch,
        leftMargin=0.42 * inch,
        topMargin=0.50 * inch,
        bottomMargin=0.45 * inch,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("ReportTitle", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=24, textColor=colors.HexColor("#202A36"), alignment=TA_LEFT, spaceAfter=12)
    subtitle_style = ParagraphStyle("ReportSubtitle", parent=styles["Normal"], fontSize=10.5, textColor=colors.HexColor("#5B6573"), alignment=TA_LEFT, spaceAfter=10)
    section_style = ParagraphStyle("Section", parent=styles["Heading2"], fontSize=14, textColor=colors.HexColor("#03547C"), alignment=TA_LEFT, spaceBefore=8, spaceAfter=5)
    note_style = ParagraphStyle("Note", parent=styles["Normal"], fontSize=7.4, textColor=colors.HexColor("#5B6573"), leftIndent=8, leading=8.8, alignment=TA_LEFT)
    header_style = ParagraphStyle("TableHeader", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=6.5, leading=7.2, textColor=colors.white, alignment=TA_CENTER)
    cell_style = ParagraphStyle("TableCell", parent=styles["Normal"], fontSize=6.2, leading=7.2, alignment=TA_CENTER)
    numeric_cell_style = ParagraphStyle("TableNumericCell", parent=styles["Normal"], fontSize=6.2, leading=7.2, alignment=TA_CENTER)

    story: list[Any] = []
    logo_bytes = base64.b64decode(BARRICK_LOGO_B64)
    logo_img = Image(io.BytesIO(logo_bytes), width=2.10 * inch, height=0.38 * inch)
    logo_img.hAlign = "LEFT"
    story.append(Spacer(1, 0.30 * inch))
    story.append(logo_img)
    story.append(Spacer(1, 0.28 * inch))
    cover_title = str(report_name or "Block_Model_Studio_Automatic_Report_v1.0").strip()
    cover_title = cover_title.replace("_", " ") or "Block Model Studio Automatic Report v1.0"
    story.append(Paragraph(html.escape(cover_title), title_style))
    story.append(Paragraph(_report_model_label(model_names), subtitle_style))
    scope_bits = [
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"BLK_MODEL: {st.session_state.get('master_blk_model_scope', 'N/A')}",
        f"Year: {_master_year_label()}",
        f"Phase: {_master_phase_label()}",
        f"Destination / Ore Type: {_master_destination_label()}",
    ]
    story.append(Paragraph(" | ".join(scope_bits), subtitle_style))
    story.append(PageBreak())

    available_width = page_size[0] - doc.leftMargin - doc.rightMargin

    if items:
        rendered_table_count = 0
        for item in items:
            table_df = item["table"].copy()
            if table_df.empty:
                continue
            if rendered_table_count > 0:
                story.append(PageBreak())
            else:
                story.append(Paragraph("Evaluation and Comparison Tables", section_style))
            story.append(Paragraph(html.escape(str(item["title"])), section_style))
            rendered_table_count += 1
            table_data = _pdf_table_data(table_df, header_style, cell_style, numeric_cell_style)
            col_widths = _pdf_column_widths(table_df, available_width)
            rl_table = Table(table_data, repeatRows=1, colWidths=col_widths, hAlign="LEFT", splitByRow=1)
            rl_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#03547C")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#D9E2EC")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("LEFTPADDING", (0, 0), (-1, -1), 2.5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 2.5),
                ("TOPPADDING", (0, 0), (-1, -1), 2.1),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2.1),
            ]))
            story.append(rl_table)
            notes = item.get("notes") or []
            if notes:
                story.append(Spacer(1, 0.04 * inch))
                story.append(Paragraph("Footnotes / filters", note_style))
                for idx, note in enumerate(notes, start=1):
                    story.append(Paragraph(f"{idx}. {html.escape(str(note))}", note_style))
            story.append(Spacer(1, 0.14 * inch))

    if charts:
        rendered_chart_count = 0
        for chart in charts:
            image_bytes = chart.get("image")
            if not image_bytes:
                continue

            # Portrait report: one chart per page for maximum readability.
            story.append(PageBreak())
            if rendered_chart_count == 0:
                story.append(Paragraph("Evaluation and Comparison Charts", section_style))
            story.append(Paragraph(html.escape(str(chart["title"])), section_style))
            rendered_chart_count += 1

            reader = ImageReader(io.BytesIO(image_bytes))
            pixel_width, pixel_height = reader.getSize()
            aspect = (float(pixel_width) / float(pixel_height)) if pixel_height else 2.0

            max_width = available_width
            max_height = 8.35 * inch
            width = max_width
            height = width / aspect if aspect else max_height
            if height > max_height:
                height = max_height
                width = height * aspect

            img = Image(io.BytesIO(image_bytes), width=width, height=height)
            img.hAlign = "LEFT"
            story.append(img)
            notes = chart.get("notes") or []
            if not notes and chart.get("note"):
                notes = [str(chart.get("note"))]
            if notes:
                story.append(Spacer(1, 0.04 * inch))
                story.append(Paragraph("Footnotes / filters", note_style))
                for idx, note in enumerate(notes, start=1):
                    story.append(Paragraph(f"{idx}. {html.escape(str(note))}", note_style))
            story.append(Spacer(1, 0.08 * inch))

    def header_footer(canvas, document):
        canvas.saveState()
        width, height = page_size
        try:
            canvas.drawImage(
                ImageReader(io.BytesIO(logo_bytes)),
                document.leftMargin,
                height - 0.34 * inch,
                width=1.10 * inch,
                height=0.20 * inch,
                preserveAspectRatio=True,
                mask="auto",
            )
        except Exception:
            pass
        canvas.setStrokeColor(colors.HexColor("#A39161"))
        canvas.setLineWidth(0.8)
        canvas.line(document.leftMargin, height - 0.39 * inch, width - document.rightMargin, height - 0.39 * inch)
        canvas.setFillColor(colors.HexColor("#5B6573"))
        canvas.setFont("Helvetica", 7.5)
        canvas.drawRightString(width - document.rightMargin, 0.20 * inch, f"Page {document.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)
    output.seek(0)
    return output.getvalue()

def render_reports() -> None:
    page_header("Report Builder", "Automatically export evaluation/comparison tables with footnotes and all available charts from the active session.")
    if not st.session_state.models:
        _render_no_models_state("No models available for reporting")
        return

    raw_bundles = list(st.session_state.models.values())
    _render_master_year_filter_sidebar(raw_bundles, "reports")
    _render_master_phase_filter_sidebar(raw_bundles, "reports")
    _render_master_destination_filter_sidebar(raw_bundles, "reports")

    scoped_bundles = {name: _scoped_bundle(bundle) for name, bundle in st.session_state.models.items()}
    model_names = list(scoped_bundles.keys())

    st.markdown(
        """
        <section class="bm-report-intro">
            <div class="bm-report-intro-kicker">Automatic package</div>
            <div class="bm-report-intro-title">Build one governed report from the active workspace</div>
            <div class="bm-report-intro-text">The Excel workbook contains principal result tables and filter notes. The PDF adds the charts available from Evaluation and Comparison.</div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    default_report_name = "Block_Model_Studio_Automatic_Report_v1.0"
    report_name = st.text_input(
        "Report name",
        value=default_report_name,
        key="automatic_report_custom_name",
        help="This name will be used on the PDF cover and as the base filename for the PDF and Excel exports.",
    ).strip()
    if not report_name:
        report_name = default_report_name
    download_base = re.sub(r"[^A-Za-z0-9._-]+", "_", report_name).strip("._-") or default_report_name
    download_base = download_base[:120]
    report_title = f"{report_name.replace('_', ' ')} - {_report_model_label(model_names)}"

    items = _automatic_report_tables(scoped_bundles)
    charts = _automatic_report_charts(scoped_bundles)

    _render_dashboard_kpi_cards([
        {"label": "Models", "value": f"{len(model_names):,}", "bg": "#D9E2F3", "border": "#AEBBD0"},
        {"label": "Tables", "value": f"{len(items):,}", "bg": "#E7E6E6", "border": "#C9C9C9"},
        {"label": "Charts", "value": f"{len(charts):,}", "bg": "#EFE5A1", "border": "#D6C45A"},
        {"label": "Year scope", "value": _master_year_label(), "bg": "#E6F4EA", "border": "#70AD47"},
    ])

    preview = pd.DataFrame([
        {"Section": item.get("category", "Table"), "Table": item["title"], "Rows": len(item["table"]), "Columns": len(item["table"].columns)}
        for item in items
    ])
    st.markdown("#### Export inventory")
    if preview.empty:
        st.warning("No report tables are available under the current model scope and filters.")
    else:
        st.dataframe(preview, use_container_width=True, hide_index=True)

    excel_bytes = _report_excel_bytes(items)
    pdf_bytes = _report_pdf_bytes(items, charts, model_names, report_name)

    st.markdown("#### Download package")
    st.caption("Both files use the same active model, year, phase and destination scope shown above.")
    export_cols = st.columns(2)
    export_cols[0].download_button(
        "Download automatic Excel tables",
        data=excel_bytes,
        file_name=f"{download_base}_Tables.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )
    export_cols[1].download_button(
        "Download automatic PDF report",
        data=pdf_bytes,
        file_name=f"{download_base}.pdf",
        mime="application/pdf",
        use_container_width=True,
    )

    with st.expander("Report contents", expanded=False):
        st.markdown(f"**{report_title}**")
        st.markdown("The Excel file contains the principal evaluation/comparison result tables and their footnotes. The PDF contains those tables plus all charts from the corresponding evaluation and comparison sections.")
        if charts:
            st.markdown("**Charts included:** " + ", ".join(chart["title"] for chart in charts))


def render_about() -> None:
    """Render the application information, authorship and technical-use notice."""
    page_header(
        "About",
        "Application information, technical scope, authorship and responsible-use notice.",
    )

    st.markdown(
        """
        <style>
        .bm-about-hero {
            display: grid;
            grid-template-columns: minmax(0, 1.8fr) minmax(230px, 0.7fr);
            gap: 1.2rem;
            padding: 1.45rem;
            margin: 0.35rem 0 1.25rem 0;
            border: 1px solid rgba(0, 84, 124, 0.16);
            border-radius: 14px;
            background:
                linear-gradient(135deg, rgba(0, 84, 124, 0.10), rgba(163, 145, 97, 0.08)),
                #FFFFFF;
            box-shadow: 0 10px 28px rgba(0, 45, 67, 0.07);
        }
        .bm-about-kicker {
            color: #A39161;
            font-size: 0.76rem;
            font-weight: 800;
            letter-spacing: 0.13em;
            text-transform: uppercase;
            margin-bottom: 0.35rem;
        }
        .bm-about-title {
            color: #004967;
            font-size: clamp(1.85rem, 3vw, 2.7rem);
            line-height: 1.05;
            font-weight: 800;
            margin-bottom: 0.65rem;
        }
        .bm-about-copy {
            color: #405563;
            font-size: 1rem;
            line-height: 1.62;
            max-width: 850px;
        }
        .bm-about-meta {
            display: flex;
            flex-direction: column;
            justify-content: center;
            gap: 0.7rem;
            padding: 1rem;
            border-left: 3px solid #A39161;
            background: rgba(255, 255, 255, 0.72);
        }
        .bm-about-meta-label {
            color: #637083;
            font-size: 0.72rem;
            font-weight: 750;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }
        .bm-about-meta-value {
            color: #1F2D36;
            font-size: 0.98rem;
            font-weight: 700;
            margin-top: 0.1rem;
        }
        .bm-about-card {
            height: 100%;
            padding: 1.1rem 1.15rem;
            border: 1px solid #D9E2EC;
            border-radius: 12px;
            background: #FFFFFF;
        }
        .bm-about-card h4 {
            color: #004967;
            margin: 0 0 0.6rem 0;
        }
        .bm-about-card p,
        .bm-about-card li {
            color: #465B68;
            line-height: 1.55;
        }
        .bm-about-notice {
            margin-top: 1rem;
            padding: 1rem 1.15rem;
            border-left: 5px solid #A39161;
            background: #F8F6EF;
            color: #3D4D57;
            line-height: 1.55;
        }
        .bm-about-footer {
            margin: 1.4rem 0 0.3rem 0;
            padding-top: 0.9rem;
            border-top: 1px solid #D9E2EC;
            color: #637083;
            font-size: 0.82rem;
            text-align: center;
        }
        @media (max-width: 760px) {
            .bm-about-hero {
                grid-template-columns: 1fr;
            }
            .bm-about-meta {
                border-left: none;
                border-top: 3px solid #A39161;
            }
        }
        </style>

        <section class="bm-about-hero">
            <div>
                <div class="bm-about-kicker">Mineral Resource Management</div>
                <div class="bm-about-title">PV BlockModel Studio</div>
                <div class="bm-about-copy">
                    A Streamlit-based analytical application designed to support block-model
                    validation, mineral-resource tabulation, model comparison and governed
                    reporting for pits and stockpiles.
                </div>
            </div>
            <div class="bm-about-meta">
                <div>
                    <div class="bm-about-meta-label">Developed by</div>
                    <div class="bm-about-meta-value">Julio Solano (2026)</div>
                </div>
                <div>
                    <div class="bm-about-meta-label">Application version</div>
                    <div class="bm-about-meta-value">Version 1.0</div>
                </div>
                <div>
                    <div class="bm-about-meta-label">Platform</div>
                    <div class="bm-about-meta-value">Python · Streamlit</div>
                </div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            <div class="bm-about-card">
                <h4>Purpose and capabilities</h4>
                <ul>
                    <li>Individual block-model evaluation and validation.</li>
                    <li>Comparison of multiple models with a mandatory global-volume check.</li>
                    <li>Resource tabulation by category and destination.</li>
                    <li>Transversal filtering by model type, year, phase and destination.</li>
                    <li>Weighted-grade and contained-metal calculations.</li>
                    <li>Automated Excel and PDF reporting.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="bm-about-card">
                <h4>Author profile</h4>
                <p><strong>Julio Solano</strong></p>
                <p>
                    Geological Engineer specializing in Mineral Resource Evaluation,
                    Geostatistics, Data Science, Data Analytics, GIS and mining applications.
                </p>
                <p>
                    Developed to strengthen repeatability, traceability and analytical
                    consistency in block-model review workflows.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Technical methodology")
    methodology = pd.DataFrame(
        [
            {
                "Component": "Weighted grades",
                "Method": "Grades are weighted by positive tonnage using Σ(grade × tonnes) / Σ(tonnes).",
            },
            {
                "Component": "Au and Ag contained metal",
                "Method": "For g/t or ppm variables, ounces are calculated as Σ(grade × tonnes) / 31.10348.",
            },
            {
                "Component": "Percent variables",
                "Method": "Percent grades are accumulated by tonnes and reported as a tonnage-weighted mean.",
            },
            {
                "Component": "Model comparison",
                "Method": "Global model volume is checked before analytical comparisons, subject to the configured tolerance.",
            },
            {
                "Component": "Governance",
                "Method": "Active filters, validation results and calculation context are retained in tables and exports where applicable.",
            },
        ]
    )
    st.dataframe(
        _left_aligned_table_style(methodology),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Component": st.column_config.TextColumn(width="medium"),
            "Method": st.column_config.TextColumn(width="large"),
        },
    )

    st.markdown(
        """
        <div class="bm-about-notice">
            <strong>Technical-use notice.</strong>
            This application is an analytical decision-support tool. Its outputs do not replace
            professional geological interpretation, formal quality assurance and quality control,
            model sign-off, or review by an appropriately qualified mineral-resource professional.
            Results must be independently verified before use in official Mineral Resource,
            Mineral Reserve, planning or public-reporting processes.
        </div>

        <div class="bm-about-footer">
            © 2026 Julio Solano. All rights reserved.
        </div>
        """,
        unsafe_allow_html=True,
    )


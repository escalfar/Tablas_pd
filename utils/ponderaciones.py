import sys
from pprint import pprint

import numpy as np
import pandas as pd

from bdd_p import BaseDatos

"""
No sirvió se usan otros métodos además de este
TODO: buscar módulos que ya hagan esto
"""


def factores_de_ponderacion_porcentajes(bd: BaseDatos, ponderaciones: dict) -> pd.Series:
    df = bd.df
    mtdt_val_lab = bd.metadata['valueLabels']
    vars_pon = list(ponderaciones.keys())
    s_pon = df.groupby(vars_pon).size()
    total = s_pon.sum()
    s_pon /= total

    def fact_pond(r):
        poblacion = np.product([ponderaciones[v_p][mtdt_val_lab[v_p][r[v_p]]] for v_p in vars_pon])
        muestra = s_pon.loc[tuple([r[v_var] for v_var in vars_pon])]

        # return muestra / poblacion
        return poblacion / muestra

    # s_facpon = pd.Series(index=df.index, dtype=float)
    s_facpon = df.apply(fact_pond, axis=1)
    s_facpon.name = 'fac_pon'

    # bd.comput_v('fac_pon', fact_pond, crear=True)
    return s_facpon


if __name__ == '__main__':
    bd1 = BaseDatos('BHT Vinson 22-01-002(enero)-CR-R2.sav')
    ponderaciones1 = {
        'ciudad': {
            'Distrito Federal': .41,
            'Guadalajara': .15,
            'Monterrey': .15,
            'Mérida': .08,
            'Puebla': .13,
            'Hermosillo': .08,
        },
        'nse_amai_2020': {
            'AB': .1,
            'C+': .4,
            'C': .4,
            'C-': .5,
            'D+': .5,
            'D': .5,
        },
        # 'sexo': {
        #     'Masculino': .4,
        #     'Femenino': .6,
        # },
    }

    s_ponderacion = factores_de_ponderacion_porcentajes(bd1, ponderaciones1)

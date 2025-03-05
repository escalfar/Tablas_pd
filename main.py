#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
import re
import sys
from pprint import pprint

import pandas as pd
import numpy as np

from utils.bdd_p import BaseDatos
from utils.writer_excel import WriterExcel
from utils.funciones_basicas import print_exit, pprint_exit, percentile, simplificar_head
from utils.funciones_complejas import por_codigo, por_codigo_est, por_variables, por_codigo_mas_variable, pruebas, \
    por_codigo_bd, por_codigo_est_bd, por_variables_bd

base_sav = 'nombre_de_la_base_de_datos.sav'
xlsx_sal = 'nombre_del_archivo_de_salida.xlsx'


# Algunas funciones que son útiles y no había encontrado dónde ponerlas
def reverse_dict_lookup(dic: dict, val, ret=None):
    try:
        key = next(key for key, value in dic.items() if value == val)
    except StopIteration:
        key = ret

    return key


def revisar_caso(df: pd.DataFrame, head: [str, list], valores: [str, list], filtro: pd.Series = None) -> pd.Series:
    if not isinstance(head, list):
        head = [head]
    if not isinstance(valores, list):
        valores = [valores]
    df_tmp = df.copy()
    if filtro is not None:
        df_tmp = df_tmp.loc[filtro]

    return (df_tmp.loc[:, head] == valores).all(axis=1)


def es_vacia(valor):
    return not (valor == -1 or np.isnan(valor))


if __name__ == '__main__':
    # Estos son los métodos bássicos que se usarían en cualquier ocasión que se generaran tablas, en el medio deberían
    # todas las instrucciones necesarias para transformar los datos usando los métodos de [utils.funciones_complejas]
    # de preferencia, pero daadas las necesidades de cada proyecto, podrían crearse métodos particulares para cada uno
    #
    # bd = BaseDatos(base_sav)
    #
    # # bd = bd.obj_metadata
    # # mtdt = bd.metadata
    #
    # wrtr = WriterExcel(xlsx_sal)
    #
    # # banner1 = pd.Index(['celda'])
    # # wrtr.add_tabla_a_excel(por_codigo_bd(bd, banner1, 'p1'), nom_hoja='p1')
    #
    # wrtr.guardar_excel()
    pass

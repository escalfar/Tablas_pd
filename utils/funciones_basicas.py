import itertools
import warnings

import pandas as pd
import numpy as np

import sys
from pprint import pprint


def print_exit(*args) -> None:
    print(*args)
    sys.exit()


def pprint_exit(*args) -> None:
    pprint(*args)
    sys.exit()


def quitar_arrobas(nombre):
    if isinstance(nombre, str):
        nombre = nombre.replace('@@', '')
    # print(nombre)

    return nombre


def mandar_base_a_inicio(df):
    return df.reindex([idx for idx in df.index if '@@Base@@' in a_lista(idx)] + [idx for idx in df.index if
                                                                                 '@@Base@@' not in a_lista(idx)])


def a_lista(*args) -> list:
    """
    Se usa para convertir objetos que funcionan como listas a listas simples
    """
    list_ret = []
    for arg in args:
        if pd.api.types.is_list_like(arg):
            list_ret.extend(arg)
        else:
            list_ret.append(arg)

    return list_ret


def percentile(n):
    """
    Genera una función para obtener el percentil que se indique
    """

    def percentile_(x):
        return x.quantile(n / 100)

    percentile_.__name__ = 'percentile_%s' % n
    return percentile_


def juntar_dfs_h(df_izq: pd.DataFrame, df_der: pd.DataFrame, how: str = 'outer', left_index: bool = True,
                 right_index: bool = True, fill_na=None, *args) -> pd.DataFrame:
    """
    Junta Data Frames de manera horizontal, ambas deben tener la misma cantidad de niveles en las columnas
    Sólo se pueden juntar dos a la vez
    """
    if len(df_izq) == 0:
        df_ret = df_der
    elif len(df_der) == 0:
        df_ret = df_izq
    else:
        df_ret = pd.DataFrame.merge(df_izq, df_der, how=how, left_index=left_index, right_index=right_index, *args)
    if fill_na is not None:
        df_ret.fillna(fill_na, inplace=True)

    return df_ret


def juntar_dfs_v(*args: pd.DataFrame, fill_na=None) -> pd.DataFrame:
    """
    Junta Data Frmaes de manera vertical (según las columnas de cada una)
    Se pueden juntar varias al mismo tiempo
    TODO: Tal vez sea mejor restringirlo a sólo dos a la vez
    """

    df_ret = pd.concat(args, verify_integrity=True)
    # print('_______________________________1')
    # print(df_ret)
    # print('_______________________________2')
    if fill_na is not None:
        df_ret = df_ret.where(df_ret.notna(), fill_na)
    # print(df_ret)
    # print('_______________________________3')

    return df_ret


def simplificar_head(head: [list, str, pd.MultiIndex, pd.Index]) -> [list, int]:
    """
    Transforma el header de un objeto Index o Multiindex a una lista simple
    Si es una cadena la convierte en una lista con ese mismo elemento
    Si es una lista no le hace nada
    """
    head_ret = set()
    for h in a_lista(head):
        head_ret.update(a_lista(h))

    niveles = 1
    if pd.api.types.is_list_like(head[0]):
        niveles = len(head[0])

    return a_lista(head_ret), niveles


def reducir_df(df: pd.DataFrame, cols: [list, str, pd.MultiIndex, pd.Index] = None, rengs: pd.Series = None,
               var_a_cod: dict = None, omitir_cods_cols: [list, dict] = None) -> [pd.DataFrame, list]:
    """
    Toma la Data Frame completa y deja sólo las variables reelevantes según el header y el side
    """
    if cols is None and rengs is None:
        # raise ValueError("Se debe dar alguno de los valores [cols] o [rengs]")
        return df

    df_ret = df.copy()
    if cols is not None:
        cols, _ = simplificar_head(cols)

        if '@@Total@@' in cols:
            cols.remove('@@Total@@')

        if omitir_cods_cols is not None:
            if not isinstance(omitir_cods_cols, dict):
                omitir_cods_cols = {c: omitir_cods_cols for c in cols}
            for col, omit_cods in omitir_cods_cols.items():
                df_ret = df_ret.loc[~df_ret.loc[:, col].isin(a_lista(omit_cods))]

        # Si [cols] es sólo una cadena se obtiene una serie, si es una lista se obtiene un DataFrame de una columna
        df_ret = df_ret.loc[:, cols]
    if rengs is not None:
        df_ret = df_ret.loc[rengs]

    assert df_ret.size != 0, 'Los datos proporcionados genera una base vacia'

    if var_a_cod is not None:
        df_ret.rename(columns=var_a_cod, inplace=True)

    return df_ret


def desglosar_variables(df: pd.DataFrame, side: [list, str]) -> pd.DataFrame:
    """
    Esta función desglosa varibales de campos a una variabel por código
    """
    side = a_lista(side)

    def desg(r):
        for var in side:
            cod = r[var]
            if np.isnan(cod):
                continue
            r[cod] = 1
        return r

    df_ret = df.apply(desg, axis=1)
    df_ret.drop(side, axis=1, inplace=True)

    return df_ret


def gen_total_respuestas(df: pd.DataFrame, cods=None) -> pd.DataFrame:
    """
    Genera el renglón con el Total
    """
    df_ret = df.copy()
    if cods is not None:
        cods = itertools.product(a_lista(cods), ('@@cont@@',))
        df_ret = df_ret.loc[df_ret.index.isin(cods)]

    df_ret = df_ret.sum().to_frame().T
    df_ret.index = [('@@Total respuestas@@', '@@cont@@')]

    df_ret = df_ret.loc[(df_ret != 0).any(axis=1)]

    return df_ret


def obtener_multiindex(df: pd.DataFrame) -> [pd.DataFrame, list]:
    cols_noms = df.columns.names if df.columns.names != [None] else []
    cols_vals = df.columns.values if df.columns.names != [None] else []

    producto = []
    llenar_vacios = []
    for n, nom in enumerate(cols_noms):
        lst_tmp = []
        for val in cols_vals:
            val = a_lista(val)
            _val = val[n]
            if _val not in lst_tmp:
                lst_tmp.append(_val)
        producto.extend([[nom], lst_tmp])
        llenar_vacios.append(lst_tmp)

    orden_vacios = list(itertools.product(*llenar_vacios))
    for col in orden_vacios:
        if col not in cols_vals and col not in a_lista(cols_vals):
            df.loc[:, col] = np.nan
    df = df[orden_vacios]

    return df, producto


def agregar_multiindex_porcentajes(df: pd.DataFrame, val: str) -> pd.DataFrame:
    """
    Agrega Multiindex para los porcentajes a una tabla
    """
    # df_ret = df.reindex([idx for idx in df.index if '@@Base@@' in a_lista(idx)] + [idx for idx in df.index if
    #                                                                                '@@Base@@' not in a_lista(idx)])
    df_ret = mandar_base_a_inicio(df)
    if '@@porcentaje@@' not in df_ret and '@@porcentaje@@' not in df_ret.index.names:
        df_ret.loc[:, '@@porcentaje@@'] = '@@cont@@'
        df_ret.index.rename('@@codigos@@', inplace=True)
    if val == '@@porc@@':
        indices = [idx for idx in df_ret.index if '@@Base@@' in a_lista(idx)]
        df_ret.drop(indices, inplace=True)

    df_ret.reset_index(inplace=True)
    df_ret.loc[df_ret.loc[:, '@@porcentaje@@'] == '@@cont@@', '@@porcentaje@@'] = val
    # df_ret.loc[(df_ret.loc[:, '@@porcentaje@@'] == '@@cont@@') & (df_ret.loc[:, '@@codigos@@'] != '@@Base@@'),
    #            '@@porcentaje@@'] = val
    df_ret.set_index(['@@codigos@@', '@@porcentaje@@'], inplace=True)
    df_ret.index.rename(['@@codigos@@', '@@porcentaje@@'])

    return df_ret


def generar_porcentajes(df: pd.DataFrame) -> pd.DataFrame:
    """
    A partir de una tabla con conteos, genera los porcentajes de estos según la base
    Necesita tener un renglón llamdo [@@Base@@]
    """
    df_por = df.drop('@@est@@', level=1, errors='ignore')
    # print_exit(df_por)

    # TODO: Buscar qué pedo con este warning, a veces aparece y a veces no
    #  No encontré mucho en internet sobre cómo evitarlo ya que está muy dentro de numpy
    #  https://stackoverflow.com/questions/40659212/futurewarning-elementwise-comparison-failed-returning-scalar-but-in-the-futur
    #  https://github.com/numpy/numpy/issues/6784
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=FutureWarning)
        wrngn = df_por * 100 / df_por.loc[('@@Base@@', '@@cont@@')]
    # print(wrngn)
    df_tbl_por = agregar_multiindex_porcentajes(wrngn, '@@porc@@')
    # df_tbl_por = agregar_multiindex_porcentajes(df_por * 100 / df_por.loc[[('@@Base@@', '@@cont@@')]], '@@porc@@')
    df = mandar_base_a_inicio(df)
    # print(df)
    # print(df_tbl_por)
    df_tbl_tot = mandar_base_a_inicio(juntar_dfs_v(df, df_tbl_por, fill_na=0).sort_index())
    # print(df_tbl_tot)
    # sys.exit()

    return df_tbl_tot


def gen_df_tabla(df: pd.DataFrame, head: [pd.MultiIndex, pd.Index], side: [str, list]) -> pd.DataFrame:
    # incluir_columnas_vacias: bool = False, incluir_renglones_vacios: bool = False) -> pd.DataFrame:
    """
    Genera las tablas con los datos
    """
    head_simp, niveles_anidados = simplificar_head(head)
    df_ret = pd.DataFrame(columns=pd.MultiIndex.from_product([[]] * (niveles_anidados * 2),
                                                             names=[x for x in range(niveles_anidados * 2)]))
    if '@@Total@@' in head_simp:
        head_tot_simp, _ = simplificar_head([x for x in head if '@@Total@@' in a_lista(x)])
        head_tot_simp.remove('@@Total@@')

        # df_ret = df.drop([x for x in head_simp if x not in head_tot_simp], axis=1)
        df_ret = df.loc[:, head_tot_simp + a_lista(side)]

        if head_tot_simp:  # Multiindex
            grp_tmp = df_ret.groupby(list(reversed(head_tot_simp)))
            base = grp_tmp.size().T
            df_ret = grp_tmp.count().T
        else:  # Index sencillo
            base = df_ret.shape[0]
            df_ret = df_ret.count().to_frame()

        df_ret.loc['@@Base@@'] = base
        df_ret = agregar_multiindex_porcentajes(df_ret, '@@cont@@').fillna(0)

        df_ret, producto = obtener_multiindex(df_ret)
        df_ret.columns = pd.MultiIndex.from_product([['@@Total@@'], ['']] + producto,
                                                    names=[x for x in range(niveles_anidados * 2)])

        # head_simp.remove('@@Total@@')
        head = [x for x in head if '@@Total@@' not in a_lista(x)]

    for h in head:
        h = a_lista(h)
        h_tab = h + a_lista(side)
        grp_tmp = df.loc[:, h_tab].groupby(h)
        df_tbl_tmp = grp_tmp.count().T
        df_tbl_tmp.loc['@@Base@@'] = grp_tmp.size()
        df_tbl_tmp = agregar_multiindex_porcentajes(df_tbl_tmp, '@@cont@@')
        df_tbl_tmp.fillna(0, inplace=True)

        df_tbl_tmp, producto = obtener_multiindex(df_tbl_tmp)
        df_tbl_tmp.columns = pd.MultiIndex.from_product(producto, names=[x for x in range(niveles_anidados * 2)])

        df_ret = juntar_dfs_h(df_ret, df_tbl_tmp)

    # if not incluir_renglones_vacios:
    #     df_ret = df_ret.loc[(df_ret != 0).any(axis=1)]
    #     df_ret = df_ret.loc[df_ret.notna().any(axis=1)]
    # if not incluir_columnas_vacias:
    #     df_ret = df_ret.loc[:, (df_ret != 0).any(axis=0)]
    #     df_ret = df_ret.loc[:, df_ret.notna().any(axis=0)]

    # df_ret = ordenar_base(df_ret)

    return df_ret


def gen_netos(df: pd.DataFrame, netos: list) -> pd.DataFrame:
    """
    Gnera netos usando una lista con esta forma
    Toma en cuenta el orden, si un nivel cuenta otro, el subnivel debe estar después en la lista
    netos = [
        {'eti': 'GRAN-NETO', 'vals': {'BEBIDAS': ['Isotónicas', 'Sueros', 'Otras']}},
        {'eti': 'NETO',
        'vals': {'Isotónicas': ['Gatorade', 'Powerade', 'Jumex Sport', 46], 'Sueros': ['Electrolit', 'Suerox'],
        'Otras': '@@all@@'}},
        {'eti': 'SUB-NETO', 'vals': {'Gatorade': [1, 4, 5, 6], 'Powerade': 2, 'Jumex Sport': 9, 'Electrolit': 3,
        'Suerox': 7}},
    ]
    TODO: Buscar cómo hacerlo más amigable para el usuario en caso de ser posible
    """
    df_tmp = df.copy()
    df_neto = df.drop(df.columns, axis=1)
    neto_all = [('@@all@@', '', [])]
    cods_usados = set()

    for datos_neto in reversed(netos):
        eti_neto = datos_neto['eti']
        if eti_neto:
            eti_neto = f' ({eti_neto})'
        vals_neto = datos_neto['vals']
        for eti, cods in vals_neto.items():
            cods = a_lista(cods)
            if neto_all[-1][0] in cods:
                neto_all.append((eti, eti_neto, cods))
                continue
            cods = [c for c in cods if c in df_tmp.columns]
            filt_neto = df_tmp.loc[:, cods].notna().any(axis=1)
            df_neto.loc[filt_neto, f'{eti}{eti_neto}'] = 1
            df_tmp.loc[filt_neto, eti] = 1
            cods_usados.update(a_lista(cods, eti))

    neto_all = neto_all[1:]
    if neto_all:
        neto_all[0][-1].remove('@@all@@')
        for eti_nt, eti_neto_nt, cods_nt in neto_all:
            df_nt = df_tmp.drop(cods_usados.difference(cods_nt), axis=1)
            filt_nt = df_nt.notna().any(axis=1)
            df_neto.loc[filt_nt, f"{eti_nt}{eti_neto_nt}"] = 1
            df_tmp.loc[filt_nt, eti_nt] = 1
            cods_usados.update(df_nt.columns)
    # Se quitan los Netos vacíos
    df_neto = df_neto.loc[:, df_neto.notna().any()]

    return df_neto


def gen_df_tabla_est(df: pd.DataFrame, head: [pd.Index, pd.MultiIndex], side: str, factor: dict = None,
                     func_agg: [list, callable] = None) -> pd.DataFrame:
    # , incluir_vacios: bool = False, incluir_renglones_vacios: bool = False) -> pd.DataFrame:
    """
    Genera los renglones con los estadisticos proporcionados
    Deben ser funciones o cadenas que la función {agg} reconozca
    """
    df_est = df.copy()
    if factor is not None:
        df_est.loc[:, side] = df_est.loc[:, side].replace(factor)

    if func_agg is None:
        # func_agg = ['mean', 'std', 'sem']
        # Hay un bug si se usa lambda en las agregadas
        # func_agg = [('MEAN', np.mean), ('S.D.', np.std), ('S.E.', lambda x: np.std(x, ddof=1) / np.sqrt(np.size(x)))]
        func_agg = [np.mean, np.std, pd.DataFrame.sem]
    head_simp, niveles_anidados = simplificar_head(head)
    df_ret = pd.DataFrame(columns=pd.MultiIndex.from_product([[]] * (niveles_anidados * 2),
                                                             names=[x for x in range(niveles_anidados * 2)]))

    if '@@Total@@' in head_simp:
        head_tot_simp, _ = simplificar_head([x for x in head if '@@Total@@' in a_lista(x)])
        head_tot_simp.remove('@@Total@@')

        df_ret_tab = df.loc[:, head_tot_simp + [side]]
        df_ret_est = df_est.loc[:, head_tot_simp + [side]]

        if head_tot_simp:  # Multiindex
            df_ret_tab = df_ret_tab.groupby(head_tot_simp + [side]).size().unstack(0)
            df_ret_est = df_ret_est.groupby(head_tot_simp).agg(func_agg).T.droplevel(0)
        else:  # Index sencillo
            df_ret_tab = df_ret_tab.groupby(side).size().to_frame().rename(columns={0: side})
            df_ret_est = df_ret_est.agg(func_agg)

        df_ret_tab.loc['@@Base@@'] = df_ret_tab.sum()
        df_ret_tab = agregar_multiindex_porcentajes(df_ret_tab, '@@cont@@').fillna(0)

        df_ret_est = agregar_multiindex_porcentajes(df_ret_est, '@@est@@')

        df_ret, producto = obtener_multiindex(juntar_dfs_v(df_ret_tab, df_ret_est))
        df_ret.columns = pd.MultiIndex.from_product([['@@Total@@'], ['']] + producto,
                                                    names=[x for x in range(niveles_anidados * 2)])

        # head_simp.remove('@@Total@@')
        head = [x for x in head if '@@Total@@' not in a_lista(x)]

    for h in head:
        h = a_lista(h)
        h_tab = h + [side]
        df_tbl_tmp_tab = df.loc[:, h_tab].groupby(h_tab).size()
        for _ in h:
            df_tbl_tmp_tab = df_tbl_tmp_tab.unstack(0)

        df_tbl_tmp_tab.loc['@@Base@@'] = df_tbl_tmp_tab.sum()
        df_tbl_tmp_tab = agregar_multiindex_porcentajes(df_tbl_tmp_tab, '@@cont@@')
        df_tbl_tmp_tab.fillna(0, inplace=True)

        df_tbl_tmp_est = df_est.loc[:, h_tab].groupby(h).agg(func_agg).T.droplevel(0)
        df_tbl_tmp_est = agregar_multiindex_porcentajes(df_tbl_tmp_est, '@@est@@')

        df_tbl_tmp, producto = obtener_multiindex(juntar_dfs_v(df_tbl_tmp_tab, df_tbl_tmp_est))
        df_tbl_tmp.columns = pd.MultiIndex.from_product(producto, names=[x for x in range(niveles_anidados * 2)])

        df_ret = juntar_dfs_h(df_ret, df_tbl_tmp)

    return df_ret


def gen_netos_est(df: pd.DataFrame, netos: list) -> pd.DataFrame:
    """
    Gnera netos usando una lista con esta forma
    Toma en cuenta el orden, si un nivel cuenta otro, el subnivel debe estar después en la lista
    netos = [
        {'eti': 'GRAN-NETO', 'vals': {'BEBIDAS': ['Isotónicas', 'Sueros', 'Otras']}},
        {'eti': 'NETO',
        'vals': {'Isotónicas': ['Gatorade', 'Powerade', 'Jumex Sport', 46], 'Sueros': ['Electrolit', 'Suerox'],
        'Otras': '@@all@@'}},
        {'eti': 'SUB-NETO', 'vals': {'Gatorade': [1, 4, 5, 6], 'Powerade': 2, 'Jumex Sport': 9, 'Electrolit': 3,
        'Suerox': 7}},
    ]
    TODO: Buscar cómo hacerlo más amigable para el usuario en caso de ser posible
    """
    df_tmp = df.copy().sort_index()
    df_neto = df_tmp.drop(df_tmp.index.tolist())
    neto_all = [('@@all@@', '', [])]
    cods_usados = set()

    for datos_neto in reversed(netos):
        eti_neto = datos_neto['eti']
        if eti_neto:
            eti_neto = f' ({eti_neto})'
        vals_neto = datos_neto['vals']
        for eti, cods in vals_neto.items():
            cods = a_lista(cods)
            if neto_all[-1][0] in cods:
                neto_all.append((eti, eti_neto, cods))
                continue
            cods = [c for c in cods if c in df_tmp.index]
            filt_neto = df_tmp.loc[[(c, '@@cont@@') for c in cods]].sum().to_frame().T
            df_neto.loc[(f'{eti}{eti_neto}', '@@cont@@'), :] = filt_neto.values[0]
            df_tmp.loc[(eti, '@@cont@@'), :] = filt_neto.values[0]
            cods_usados.update(a_lista(cods, eti))

    neto_all = neto_all[1:]
    if neto_all:
        neto_all[0][-1].remove('@@all@@')
        for eti_nt, eti_neto_nt, cods_nt in neto_all:
            df_nt = df_tmp.drop(cods_usados.difference(cods_nt), level=0)
            filt_nt = df_nt.notna().any(axis=1)
            df_neto.loc[filt_nt, f"{eti_nt}{eti_neto_nt}"] = 1
            df_tmp.loc[filt_nt, eti_nt] = 1
            cods_usados.update(df_nt.columns)
    # Se quitan los Netos vacíos
    df_neto = df_neto.loc[:, df_neto.notna().any()]

    return df_neto


def cambiar_nombres_ejes(df: pd.DataFrame, cods_head: dict = None, cods_side: dict = None,
                         nom_vars: dict = None) -> pd.DataFrame:
    """
    Cambia los nombres de las columnas y los renglones
    """
    if cods_head is None and cods_side is None and nom_vars is None:
        # raise ValueError("Se debe dar alguno de los valores [cods_head] o [cods_side]")
        return df

    df_ret = df.copy()
    if cods_head is not None:
        n_mltindx = []
        for idx in df.columns:
            n_idx = []
            for i, var in enumerate(idx[::2]):
                n_idx.extend([var, cods_head.get(var, {}).get(idx[2 * i + 1], idx[2 * i + 1])])
            n_mltindx.append(tuple(n_idx))
        df_ret.columns = pd.MultiIndex.from_tuples(n_mltindx, names=df.columns.names)

    if nom_vars is not None:
        df_ret.rename(columns=nom_vars, inplace=True)

    if cods_side is not None:
        df_ret.rename(index=cods_side, inplace=True, level=0)

    df_ret.rename(columns=quitar_arrobas, index=quitar_arrobas, inplace=True)
    df_ret.index.rename([quitar_arrobas(idx) for idx in df_ret.index.names], inplace=True)

    return df_ret

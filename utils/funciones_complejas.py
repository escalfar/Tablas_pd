import sys
from pprint import pprint

from utils.funciones_basicas import *
import pandas as pd
import numpy as np
from bdd_p import BaseDatos


def pruebas(df, head, side, filtro, fac_conv_est):
    pass
    # # if '@@Total@@' in head:  # TODO: Modificar con multiindex
    # #     # incluir_total = True
    # #     head.remove('@@Total@@')
    # df_head = reducir_df(df, cols=head, rengs=filtro)
    # df_side = desglosar_variables(df_side, side, omitir_codigos=omitir_codigos_side)
    # # df_side = reducir_df(df, cols=side, rengs=filtro)
    # df_tmp = juntar_dfs_h(df_head, df_side)
    #
    # for dem in head:
    #     df_tmp_ = df_tmp.drop([d for d in head if d != dem], axis=1)
    #     df_tmp_.loc[:, side] = df_tmp_.loc[:, side].replace(fac_conv_est)
    #     grp_tmp = df_tmp_.groupby(dem)
    #
    #     print_exit(
    #         grp_tmp.agg(['mean', 'std', 'sem'], axis=1).T)
    #     sys.exit()
    # # print(df_side)
    # sys.exit()


def por_codigo(df: pd.DataFrame, head: [pd.Index, pd.MultiIndex], side: [list, str], cods_head: dict = None,
               cods_side: dict = None, filtro: pd.Series = None, omitir_codigos_head: [dict, list] = None,
               omitir_codigos_side: list = None, netos: list = None, nom_vars: dict = None) -> pd.DataFrame:
    # , incluir_columnas_vacias: bool = False, incluir_renglones_vacios: bool = False) -> pd.DataFrame:
    # TODO: El side aún no está listo para pasarle listas, arreglar. Hay que meter MultiIndex :(
    df_side = reducir_df(df, cols=side, rengs=filtro, omitir_cods_cols=omitir_codigos_side)
    df_side = desglosar_variables(df_side, side)

    if netos is not None:
        df_netos = gen_netos(df=df_side, netos=netos)
        df_side = juntar_dfs_h(df_side, df_netos)
    side = df_side.columns
    df_head = reducir_df(df, cols=head, rengs=filtro, omitir_cods_cols=omitir_codigos_head)

    df_ret = gen_df_tabla(juntar_dfs_h(df_head, df_side), head, side)
    # , incluir_columnas_vacias=incluir_columnas_vacias, incluir_renglones_vacios=incluir_renglones_vacios)

    df_ret = juntar_dfs_v(df_ret, gen_total_respuestas(df=df_ret, cods=side), fill_na=0)
    # print_exit(df_ret.to_string())
    df_ret = generar_porcentajes(df_ret)
    # print(df_ret.to_string())
    # sys.exit()
    df_ret = cambiar_nombres_ejes(df_ret, cods_head=cods_head, cods_side=cods_side, nom_vars=nom_vars)

    return df_ret


def por_codigo_bd(bd: BaseDatos, head: [pd.Index, pd.MultiIndex], side: [list, str], cods_head: dict = None,
                  cods_side: dict = None, filtro: pd.Series = None, omitir_codigos_head: [dict, list] = None,
                  omitir_codigos_side: list = None, netos: list = None, nom_vars: dict = None) -> pd.DataFrame:
    # TODO: El side aún no está listo para pasarle listas, arreglar. Hay que meter MultiIndex :(
    head_simp, _ = simplificar_head(head)
    side_lst = a_lista(side)

    if cods_head is None:
        cods_head = {variable: bd.get_val_eti(variable) for variable in head_simp}
    if cods_side is None:
        cods_side = bd.get_val_eti(side)
    if nom_vars is None:
        nom_vars = {variable: bd.get_var_eti(variable) for variable in head_simp + side_lst}

    df_ret = por_codigo(bd.df, head, side, cods_head, cods_side, filtro, omitir_codigos_head, omitir_codigos_side,
                        netos, nom_vars)
    return df_ret


def por_variables(df: pd.DataFrame, head: [pd.Index, pd.MultiIndex], side: [list, str], var_a_cod: dict = None,
                  cods_head: dict = None, cods_side: dict = None, filtro: pd.Series = None,
                  omitir_codigos_head: list = None, netos: list = None, nom_vars: dict = None) -> pd.DataFrame:
    # , incluir_columnas_vacias: bool = False, incluir_renglones_vacios: bool = False) -> pd.DataFrame:
    df_side = reducir_df(df, cols=side, rengs=filtro, var_a_cod=var_a_cod)
    side = df_side.columns
    # Se elimina todos los que no sean 1
    df_side.loc[:, side] = df_side.loc[:, side].where(df_side.loc[:, side].isin([1]), np.nan)
    if netos is not None:
        df_netos = gen_netos(df=df_side, netos=netos)
        df_side = juntar_dfs_h(df_side, df_netos)
    side = df_side.columns

    df_head = reducir_df(df, cols=head, rengs=filtro, omitir_cods_cols=omitir_codigos_head)
    df_ret = juntar_dfs_h(df_head, df_side)

    df_ret = gen_df_tabla(df_ret, head, side)
    # , incluir_columnas_vacias=incluir_columnas_vacias, incluir_renglones_vacios=incluir_renglones_vacios)
    df_ret = juntar_dfs_v(df_ret, gen_total_respuestas(df=df_ret, cods=side), fill_na=0)
    df_ret = generar_porcentajes(df_ret)
    df_ret = cambiar_nombres_ejes(df_ret, cods_head=cods_head, cods_side=cods_side, nom_vars=nom_vars)

    return df_ret


def por_variables_bd(bd: BaseDatos, head: [pd.Index, pd.MultiIndex], side: [list, str], var_a_cod: dict = None,
                     cods_head: dict = None, cods_side: dict = None, filtro: pd.Series = None,
                     omitir_codigos_head: list = None, netos: list = None, nom_vars: dict = None) -> pd.DataFrame:
    head_simp, _ = simplificar_head(head)
    side_lst = a_lista(side)

    if var_a_cod is None:
        var_a_cod = {variable: float(variable.split('_')[-1]) for variable in side_lst}
    if cods_head is None:
        cods_head = {variable: bd.get_val_eti(variable) for variable in head_simp}
    if cods_side is None:
        cods_side = {float(variable.split('_')[-1]): bd.get_var_eti(variable).split('(')[-1][:-1] for variable in
                     side_lst}
    if nom_vars is None:
        nom_vars = {variable: bd.get_var_eti(variable) for variable in head_simp + side_lst}

    df_ret = por_variables(bd.df, head, side, var_a_cod, cods_head, cods_side, filtro, omitir_codigos_head, netos,
                           nom_vars)
    return df_ret


def por_codigo_mas_variable(df: pd.DataFrame, head: [pd.Index, pd.MultiIndex], side: [list, str], side_var: [list, str],
                            var_a_cod: dict, cods_head: dict = None, cods_side: dict = None, filtro: pd.Series = None,
                            omitir_codigos_head: list = None, omitir_codigos_side: list = None, netos: list = None,
                            nom_vars: dict = None) -> pd.DataFrame:
    # , incluir_columnas_vacias: bool = False, incluir_renglones_vacios: bool = False) -> pd.DataFrame:
    # TODO: El side aún no está listo para pasarle listas en la aprte de códigos, arreglar. Hay que meter MultiIndex :(
    df_side_cod = reducir_df(df, cols=side, rengs=filtro, omitir_cods_cols=omitir_codigos_side)
    df_side_cod = desglosar_variables(df_side_cod, side)

    df_side_var = reducir_df(df, cols=side_var, rengs=filtro, var_a_cod=var_a_cod)
    side_var = df_side_var.columns
    # Se elimina todos los que no sean 1
    df_side_var.loc[:, side_var] = df_side_var.loc[:, side_var].where(df_side_var.loc[:, side_var].isin([1]), np.nan)

    df_side = pd.DataFrame.add(df_side_cod, df_side_var, fill_value=0)  # Se suman ambos DataFrames
    df_side = df_side.where(df_side != 0, np.nan)  # Se convierten los 0 a NaN
    df_side = df_side.where(df_side.isna(), 1)  # Se convierte todos lo que no sea NaN a 1

    if netos is not None:
        df_netos = gen_netos(df=df_side, netos=netos)
        df_side = juntar_dfs_h(df_side, df_netos)
    side = df_side.columns

    df_head = reducir_df(df, cols=head, rengs=filtro, omitir_cods_cols=omitir_codigos_head)
    df_ret = juntar_dfs_h(df_head, df_side)

    df_ret = gen_df_tabla(df_ret, head, side)
    # , incluir_columnas_vacias=incluir_columnas_vacias, incluir_renglones_vacios=incluir_renglones_vacios)
    df_ret = juntar_dfs_v(df_ret, gen_total_respuestas(df=df_ret, cods=side), fill_na=0)
    df_ret = generar_porcentajes(df_ret)
    df_ret = cambiar_nombres_ejes(df_ret, cods_head=cods_head, cods_side=cods_side, nom_vars=nom_vars)

    return df_ret


def por_codigo_est(df: pd.DataFrame, head: [pd.Index, pd.MultiIndex], side: str, cods_head: dict = None,
                   cods_side: dict = None, filtro: pd.Series = None, omitir_codigos_head: [dict, list] = None,
                   omitir_codigos_side: list = None, netos: list = None, factor: dict = None,
                   nom_vars: dict = None) -> pd.DataFrame:
    # , incluir_columnas_vacias: bool = False, incluir_renglones_vacios: bool = False) -> pd.DataFrame:
    # TODO: El side aún no está listo para pasarle listas, arreglar. Hay que meter MultiIndex :(
    df_side = reducir_df(df, cols=side, rengs=filtro, omitir_cods_cols=omitir_codigos_side)

    df_head = reducir_df(df, cols=head, rengs=filtro, omitir_cods_cols=omitir_codigos_head)

    df_ret = gen_df_tabla_est(df=juntar_dfs_h(df_head, df_side), head=head, side=side, factor=factor)
    # ,incluir_columnas_vacias=incluir_columnas_vacias, incluir_renglones_vacios=incluir_renglones_vacios)
    if netos is not None:
        df_netos = gen_netos_est(df=df_ret, netos=netos)
        df_ret = juntar_dfs_v(df_ret, df_netos, fill_na=0)

    df_ret = juntar_dfs_v(df_ret, gen_total_respuestas(df=df_ret, cods=df.loc[:, side].unique()), fill_na=0)
    df_ret = generar_porcentajes(df_ret)
    df_ret = cambiar_nombres_ejes(df_ret, cods_head=cods_head, cods_side=cods_side, nom_vars=nom_vars)

    return df_ret


def por_codigo_est_bd(bd: BaseDatos, head: [pd.Index, pd.MultiIndex], side: str, cods_head: dict = None,
                      cods_side: dict = None, filtro: pd.Series = None, omitir_codigos_head: [dict, list] = None,
                      omitir_codigos_side: list = None, netos: list = None, factor: dict = None,
                      nom_vars: dict = None) -> pd.DataFrame:
    # TODO: El side aún no está listo para pasarle listas, arreglar. Hay que meter MultiIndex :(
    head_simp, _ = simplificar_head(head)
    side_lst = a_lista(side)

    if cods_head is None:
        cods_head = {variable: bd.get_val_eti(variable) for variable in head_simp}
    if cods_side is None:
        cods_side = bd.get_val_eti(side)
    if nom_vars is None:
        nom_vars = {variable: bd.get_var_eti(variable) for variable in head_simp + side_lst}

    df_ret = por_codigo_est(bd.df, head, side, cods_head, cods_side, filtro, omitir_codigos_head, omitir_codigos_side,
                            netos, factor, nom_vars)
    return df_ret


def por_variables_est(df: pd.DataFrame, head: [pd.Index, pd.MultiIndex], side: [list, str], var_a_cod: dict = None,
                      cods_head: dict = None, cods_side: dict = None, filtro: pd.Series = None,
                      omitir_codigos_head: list = None, netos: list = None, nom_vars: dict = None) -> pd.DataFrame:
    # pass
    return pd.DataFrame()

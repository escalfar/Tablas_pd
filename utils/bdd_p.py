#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
from copy import deepcopy
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from pyreadstat import read_sav, write_sav, metadata_container

# TODO: Arcutalizar a 3.00 cuando esté lista
# Versión 2.11


class ErrBDD(Exception):
    pass


class BaseDatosVacia:
    """
    Crea un objeto BaseDatos sin datos y sin necesidad de incluir un archivo
    """

    def __init__(self, nombre: str, _imp: bool = True) -> None:
        self.nombre = nombre

        self.df = pd.DataFrame()
        self.metadata = metadata_container

        self.vars_base = []

        self.eti_vars = {}
        self.eti_values = {}
        self.medidas_vars = {}
        self.formatos_vars = {}

        self._atribs_var = (self.eti_vars, self.eti_values, self.medidas_vars, self.formatos_vars)

        if _imp:
            print(f'Base [{self.nombre}] creada.')

    def __str__(self):
        return self.df.to_string()

    @property
    def atribs_var(self):
        return self._atribs_var

    @property
    def nombre(self):
        return self._nombre

    @nombre.setter
    def nombre(self, nom):
        self._nombre = nom

    def vars(self, *rangos) -> list:
        """
        Devuelve una lista con las variables en los rangos proporcionados
        Si no se incluyen rangos se devuleve la lista completa de variables
        """

        vars_base_ = self.vars_base

        if not rangos:
            return vars_base_

        l_vars = []

        for rango in rangos:
            if isinstance(rango, str):
                if rango not in vars_base_:
                    raise ErrBDD(f'No existe [{rango}] en la base de datos.')
                l_vars.append(rango)
            else:
                msj_err = None
                ind_ini, ind_fin = None, None
                try:
                    ind_ini, ind_fin = tuple(vars_base_.index(nvar) for nvar in rango)
                except ValueError as e:
                    msj_err = str(e)
                else:
                    if ind_ini >= ind_fin:
                        msj_err = f'Error en orden de rango de variables: [{rango}]'

                if msj_err:
                    raise ErrBDD(f'Grupo mal especificado: {msj_err}.')

                l_vars.extend(vars_base_[ind_ini: ind_fin + 1])

        duplicados = [item for item, count in Counter(l_vars).items() if count > 1]

        if duplicados:
            raise ErrBDD(f'Variables duplicadas: {tuple(duplicados)}.')

        return l_vars

    def get_var_eti(self, nombre: str) -> str:
        """
        Regresa la etiqueta de una variable
        """
        if nombre not in self.vars_base:
            raise ErrBDD(f'No existe [{nombre}] en la base de datos.')

        return self.eti_vars[nombre]

    def set_var_eti(self, l_var: [str, list, tuple], etiqueta: str) -> None:
        """
        Modifica la etiqueta de una o varias variables
        """
        if isinstance(l_var, str):
            l_var = self.vars(l_var)
        else:
            l_var = self.vars(*l_var)

        if callable(etiqueta):
            callback = etiqueta
        else:
            def callback(*args):
                return etiqueta

        for ind, nvar in enumerate(l_var):
            eti_o = self.eti_vars[nvar]
            self.eti_vars[nvar] = callback(nvar, eti_o, ind)

    def get_val_eti(self, nombre: str) -> dict:
        """
        Regresa las etiquetas de los valores de una variable
        """
        if nombre not in self.vars_base:
            raise ErrBDD(f'No existe [{nombre}] en la base de datos.')

        return self.eti_values[nombre]

    def set_val_eti(self, l_var: [str, list, tuple], etiquetas: dict[[int, float], str]) -> None:
        """
        Modifica las etiquetas de los valores de una o varias variables
        """
        if isinstance(l_var, str):
            l_var = self.vars(l_var)
        else:
            l_var = self.vars(*l_var)

        if callable(etiquetas):
            callback = etiquetas
        else:
            def callback(*args):
                return etiquetas

        for ind, nvar in enumerate(l_var):
            etis_o = self.eti_values[nvar]
            self.eti_values[nvar] = callback(nvar, etis_o, ind)

    def get_frmt(self, nombre: str) -> str:
        """
        Regresa el formato de una variable
        """
        if nombre not in self.vars_base:
            raise ErrBDD(f'No existe [{nombre}] en la base de datos.')

        return self.formatos_vars[nombre]

    def set_frmt(self, l_var: [str, list, tuple], formatos: [callable, str]) -> None:
        """
        Mofica el formato de una o varias variables
        """
        if isinstance(l_var, str):
            l_var = self.vars(l_var)
        else:
            l_var = self.vars(*l_var)

        if callable(formatos):
            callback = formatos
        else:
            def callback(*args):
                return formatos

        for ind, nvar in enumerate(l_var):
            frms = self.formatos_vars[nvar]
            self.formatos_vars[nvar] = callback(nvar, frms, ind)

    def renombrar_var(self, l_var: [str, list, tuple], nombre: [str, callable]) -> None:
        """
        Renombra una variable en la metadata y el DataFrame
        """
        if isinstance(l_var, str):
            l_var = self.vars(l_var)
        else:
            l_var = self.vars(*l_var)

        if callable(nombre):
            callback = nombre
        else:
            def callback(*args):
                return nombre

        for ind, nvar in enumerate(l_var):
            ind_var = self.vars_base.index(nvar)
            eti_o = self.eti_vars[nvar]
            nuevo_nvar = callback(nvar, eti_o, ind)

            self.vars_base[ind_var] = nuevo_nvar

            # Modificar metadatos
            # eti_vars, eti_values, medidas_vars, formatos_vars
            # for self_dic in (self.eti_vars, self.eti_values, self.medidas_vars, self.formatos_vars):
            for self_dic in self.atribs_var:
                self_dic[nuevo_nvar] = self_dic.pop(nvar)

            self.df.rename(columns={nvar: nuevo_nvar}, inplace=True)

        duplicados = [elem for elem, cuenta in Counter(self.vars_base).items() if cuenta > 1]

        if duplicados:
            raise ErrBDD(f'Variables duplicadas: [{tuple(duplicados)}].')

    def incluir_vars(self, l_var: [str, list, tuple], elim: bool = False, orden_base: bool = False,
                     todas: bool = False) -> None:
        """
        Solo deja las variables listadas
        """
        if elim and (todas or len(l_var) == 0):
            raise ErrBDD('Se eliminarían todas las variables de la base.')

        if isinstance(l_var, str):
            l_var = self.vars(l_var)
        else:
            l_var = self.vars(*l_var)

        if elim:
            l_var = [nv for nv in self.vars_base if nv not in l_var]

        if orden_base:
            l_var = [nv for nv in self.vars_base if nv in l_var]

        if todas:
            l_var += [nv for nv in self.vars_base if nv not in l_var]

        # Modificar metadatos
        # Aquí sólo se quitan las variables de los atributos
        # eti_vars, eti_values, medidas_vars, formatos_vars
        for self_dic in self.atribs_var:
            [self_dic.pop(var, None) for var in self.vars_base if var not in l_var]

        self.df = self.df.loc[:, l_var]

        self.vars_base = l_var

    def borrar_vars(self, l_var: [str, list, tuple]) -> None:
        """
        Elimina las variables listadas
        """
        if not len(l_var) == 0:
            self.incluir_vars(l_var, elim=True)

    def crear_var(self, nombre: str, antes: str = None, despues: str = None, medida: str = 'nominal', eti: str = None,
                  fmt: str = 'F3', eti_vals: dict[[int, float], str] = None) -> None:
        """
        Crea una nueva variable
        """
        if nombre in self.vars_base:
            raise ErrBDD(f'Ya existe la variable [{nombre}]')

        if antes and despues:
            raise ErrBDD('Sólo debe especificarse un parámetro dentro de [antes, despues].')
        elif antes or despues:
            err_msj = None
            if antes is not None and antes not in self.vars_base:
                err_msj = antes
            if despues is not None and despues not in self.vars_base:
                err_msj = despues

            if err_msj:
                raise ErrBDD(f'No existe la variable [{err_msj}].')
            pos = antes or despues
        else:
            pos = self.vars_base[-1]

        # Índice de inserción
        indice = self.vars_base.index(pos)
        if despues or not antes:
            indice += 1

        # Agregar en datos
        self.df.insert(indice, nombre, np.nan)

        # eti_vars, eti_values, medidas_vars, formatos_vars
        self.eti_vars[nombre] = eti if eti is not None else nombre
        self.eti_values[nombre] = eti_vals if eti_vals is not None else {}
        self.medidas_vars[nombre] = medida
        self.formatos_vars[nombre] = fmt

        # Agregar en variables
        self.vars_base.insert(indice, nombre)

    def comput_var(self, nombre: str, valor, crear: bool = False) -> None:
        """
        Calcula un valor y lo asigna a la variable
        """
        if crear:
            self.crear_var(nombre)

        if nombre not in self.vars_base:
            raise ErrBDD(f'No existe la variable [{nombre}].')

        if callable(valor):
            callback = valor
        else:
            def callback(r):
                return valor

        self.df.loc[:, nombre] = self.df.apply(callback, axis=1)

    def recod_var(self, nombre: [str, list, tuple], rec: dict[[int, float], [int, float]]) -> None:
        """
        Recodifica una variable
        """
        if isinstance(nombre, str):
            nombres = self.vars(nombre)
        else:
            nombres = self.vars(*nombre)

        if callable(rec):
            callback = rec
        else:
            def callback(r):
                return rec

        listas = zip(*callback(self.df).items())
        df_tmp = self.df.loc[:, nombres]
        df_tmp.replace(*listas, inplace=True)

        self.df.loc[:, nombres] = df_tmp

    def guardar_bdd(self, ruta: str, forzar: bool = False) -> None:
        """
        Guarda la base de datos en disco
        """
        if not forzar:
            if Path(ruta).is_file():
                raise ErrBDD(f"El archivo '{ruta}' ya existe.")
        else:
            try:
                with open(ruta, 'w') as _:
                    pass
            except PermissionError:
                raise ErrBDD(f"El archivo '{ruta}' esta siendo usado por otro programa, no se puede guardar.")

        write_sav(self.df, ruta, column_labels=self.eti_vars, variable_value_labels=self.eti_values,
                  variable_measure=self.medidas_vars, variable_format=self.formatos_vars, compress=True)
        """
        Parámetros que se le pueden pasar a write_sav
        df : pandas data frame
            pandas data frame to write to sav or zsav
        dst_path : str or pathlib.Path
            full path to the result sav or zsav file
        file_label : str, optional
            a label for the file
        column_labels : list or dict, optional
            labels for columns (variables), if list must be the same length as the number of columns. Variables with no
            labels must be represented by None. If dict values must be variable names and values variable labels.
            In such case there is no need to include all variables; labels for non existent
            variables will be ignored with no warning or error.
        compress : boolean, optional
            if true a zsav will be written, by default False, a sav is written
        row_compress : boolean, optional
            if true it applies row compression, by default False, compress and row_compress cannot be both true at the
            same time
        note : str, optional
            a note to add to the file
        variable_value_labels : dict, optional
            value labels, a dictionary with key variable name and value a dictionary with key values and
            values labels. Variable names must match variable names in the dataframe otherwise will be
            ignored. Value types must match the type of the column in the dataframe.
        missing_ranges : dict, optional
            user defined missing values. Must be a dictionary with keys as variable names matching variable
            names in the dataframe. The values must be a list. Each element in that list can either be
            either a discrete numeric or string value (max 3 per variable) or a dictionary with keys 'hi' and 'lo' to
            indicate the upper and lower range for numeric values (max 1 range value + 1 discrete value per
            variable). hi and lo may also be the same value in which case it will be interpreted as a discrete
            missing value.
            For this to be effective, values in the dataframe must be the same as reported here and not NaN.
        variable_display_width : dict, optional
            set the display width for variables. Must be a dictonary with keys being variable names and
            values being integers.
        variable_measure: dict, optional
            sets the measure type for a variable. Must be a dictionary with keys being variable names and
            values being strings one of "nominal", "ordinal", "scale" or "unknown" (default).
        variable_format: dict, optional
            sets the format of a variable. Must be a dictionary with keys being the variable names and 
            values being strings defining the format. See README, setting variable formats section,
            for more information.
        """

        print(f'Guardada en [{ruta}].')


class BaseDatos(BaseDatosVacia):
    """
    Base de datos

    """

    def __init__(self, ruta: str, nombre: str = None, encoding: str = None) -> None:
        self.ruta = ruta
        if nombre is None:
            nombre = os.path.split(self.ruta)[1].split('.')[0]

        super(BaseDatos, self).__init__(nombre=nombre, _imp=False)

        if encoding is None:
            # También puede ser "L1", "latin1" o "ISO-8859-1", los tres para iso 8859-1
            encoding = 'UTF-8'

        self.df, self.metadata = read_sav(ruta, encoding=encoding)
        """
        Parametros de self.metadata
        notes:
            notes or documents (text annotations) attached to the file if any (spss and stata).
        column_names:
            a list with the names of the columns.
        column_labels:
            a list with the column labels, if any.
        column_names_to_labels:
            a dictionary with column_names as keys and column_labels as values
        file_encoding:
            a string with the file encoding, may be empty
        number_columns:
            an int with the number of columns
        number_rows:
            an int with the number of rows. If metadataonly option was used, it may be None if the number of rows could
            not be determined. If you need the number of rows in this case you need to parse the whole file.
        variable_value_labels:
            a dict with keys being variable names, and values being a dict with values as keys and labels as values.
            It may be empty if the dataset did not contain such labels. For sas7bdat files it will be empty unless a
            sas7bcat was given. It is a combination of value_labels and variable_to_label.
        value_labels:
            a dict with label name as key and a dict as value, with values as keys and labels as values. In the case of
            parsing a sas7bcat file this is where the formats are.
        variable_to_label:
            A dict with variable name as key and label name as value. Label names are those described in value_labels.
            Sas7bdat files may have this member populated and its information can be used to match the information in
            the value_labels coming from the sas7bcat file.
        original_variable_types:
            a dict of variable name to variable format in the original file. For debugging purposes.
        readstat_variable_types:
            a dict of variable name to variable type in the original file as extracted by Readstat.i For debugging
            purposes. In SAS and SPSS variables will be either double (numeric in the original app)
            or string (character). Stata has in addition int8, int32 and float types.
        table_name:
            table name (string)
        file_label:
            file label (SAS) (string)
        missing_ranges:
            a dict with keys being variable names. Values are a list of dicts. Each dict contains two keys,
            ‘lo’ and ‘hi’ being the lower boundary and higher boundary for the missing range. Even if the value in both
            lo and hi are the same, the two elements will always be present. This appears for SPSS (sav) files when
            using the option user_missing=True: user defined missing values appear not as nan but as their true value
            and this dictionary stores the information about which values are to be considered missing.
        missing_user_values:
            a dict with keys being variable names. Values are a list of character values (A to Z and _ for SAS,
            a to z for SATA) representing user defined missing values in SAS and STATA. This appears when using
            user_missing=True in read_sas7bdat or read_dta if user defined missing values are present.
        variable_alignment:
            a dict with keys being variable names and values being the display alignment: left, center, right or unknown
        variable_storage_width:
            a dict with keys being variable names and values being the storage width
        variable_display_width:
            a dict with keys being variable names and values being the display width
        variable_measure:
            a dict with keys being variable names and values being the measure: nominal, ordinal, scale or unknown
        """

        # Variables de la base
        self.vars_base = self.metadata.column_names
        # self.vars_base.extend(self.metadata.column_names)

        # Etiqueta de las variables
        # self.eti_vars = {var: var for var in self.vars_base}  # Si no tiene etiqueta se pone el nombre de la variable
        self.eti_vars.update(
            {var: var for var in self.vars_base})  # Si no tiene etiqueta se pone el nombre de la variable
        self.eti_vars.update(self.metadata.column_names_to_labels)
        # Etiquetas de los códigos
        # self.eti_values = {var: {} for var in self.vars_base}
        self.eti_values.update({var: {} for var in self.vars_base})
        self.eti_values.update(self.metadata.variable_value_labels)
        # Tamaño de las variables
        # self.medidas_vars = self.metadata.variable_measure
        self.medidas_vars.update(self.metadata.variable_measure)
        # Formato de las variables
        # self.formatos_vars = self.metadata.original_variable_types
        self.formatos_vars.update(self.metadata.original_variable_types)

        print(f'Base [{self.nombre}] cargada.')


def juntar_metadata(bd_1: [BaseDatos, BaseDatosVacia], bd_2: [BaseDatos, BaseDatosVacia],
                    nombre: str = None) -> BaseDatos:
    """
    Hace una copia de objeto [bd_1], y agrega la metadata de [bd_2]. Devuelve un obejto BaseDatos con un Dataframe vacío
    Importante: Si hay variables con metadata en las dos bases, se sobreescribirá los de [bd_2] con los de [bd_1]
    TODO: Agregar una opción para escoger qué metadata se queda
    """
    # Chequeos de bases vacías
    if not isinstance(bd_1, BaseDatos) and not isinstance(bd_2, BaseDatos):
        raise ErrBDD("Se está intentando juntar la metadata de dos bases vacías.")
    if not isinstance(bd_1, BaseDatos):
        return bd_2
    if not isinstance(bd_2, BaseDatos):
        return bd_1

    bd_ret = deepcopy(bd_1)
    if nombre is None:
        nombre = f'{bd_1.nombre} + {bd_2.nombre}'
    bd_ret.nombre = nombre

    # Se agregan las variables nuevas a la lista de la metadata
    # vars_2 = [nv for nv in bd_2.vars_base if nv not in bd_ret.vars_base]
    # Se divide la lista de variables a renombrar entre las que están en [bd_ret] y las que no
    vars_en_ret, vars_no_en_ret = [], []
    for x in bd_2.vars_base:
        (vars_no_en_ret, vars_en_ret)[x in bd_ret.vars()].append(x)
    
    # bd_ret.vars_base = bd_ret.vars_base + vars_2
    bd_ret.vars_base = bd_ret.vars_base + vars_no_en_ret

    # No estoy seguro si sirve de algo, pero lo hago para copiar los tipos de las columnas nuevas
    # bd_ret.df[vars_2] = bd_2.df.loc[:, vars_2]
    # bd_ret.df[vars_2] = bd_2.df[vars_2]
    bd_ret.df = pd.merge(bd_ret.df, bd_2.df.drop(vars_en_ret, axis=1), how='outer', left_index=True, right_index=True)
    # Se eliminan todos los datos y solo se dejan las columnas con sus nombres
    bd_ret.df = bd_ret.df.iloc[0:0]

    # Aquí se va por cada uno de los diccionarios que se ocupan al guardar la base
    for dic_ret, dic_2 in zip(bd_ret.atribs_var, bd_2.atribs_var):
        for nv in dic_2:
            if nv not in dic_ret:
                dic_ret[nv] = dic_2[nv]

    return bd_ret


def agregar_variables(base_izq: [BaseDatos, BaseDatosVacia], base_der: [BaseDatos, BaseDatosVacia],
                      identificador: [str, list, tuple], nombre: str = None, imp: bool = True) -> BaseDatos:
    """
    Junta dos bases con el mismo identificador pero con diferencias en las columnas
    Importante: Si hay variables repetidas en las dos bases, se sobreescribirán los de [base_der] con los de [base_izq]
    TODO: Agregar una opción para escoger qué datos y metadata se quedan
    """
    # Chequeos de bases vacías
    if not isinstance(base_izq, BaseDatos) and not isinstance(base_der, BaseDatos):
        raise ErrBDD("Se está intentando juntar dos bases vacías.")
    if not isinstance(base_izq, BaseDatos):
        return base_der
    if not isinstance(base_der, BaseDatos):
        return base_izq

    if isinstance(identificador, str):
        identificador = [identificador]
    bd_ret = juntar_metadata(base_izq, base_der, nombre=nombre)
    # Se obtienen las variables compartidas para eliminarlas de [base_der]
    vars_repe = [nv for nv in base_izq.vars_base if nv in base_der.vars_base and nv not in identificador]
    bd_ret.df = pd.merge(base_izq.df, base_der.df.drop(vars_repe, axis=1), how='outer', on=identificador,
                         validate='1:1')

    # Imprime las diferencias en las variables
    vars_izq_no_der = [nv for nv in base_izq.vars_base if nv not in base_der.vars_base]
    if vars_izq_no_der and imp:
        print(f'\nVariables en [{base_izq.nombre}] que no están en [{base_der.nombre}]:')
        for v in vars_izq_no_der:
            print(f'\t{v}')

    vars_der_no_izq = [nv for nv in base_der.vars_base if nv not in base_izq.vars_base]
    if vars_der_no_izq and imp:
        print(f'\nVariables en [{base_der.nombre}] que no están en [{base_izq.nombre}]:')
        for v in vars_der_no_izq:
            print(f'\t{v}')
    print()
    print(f'Base [{bd_ret.nombre}] creada')

    return bd_ret


def agregar_casos(base_arr: [BaseDatos, BaseDatosVacia], base_aba: [BaseDatos, BaseDatosVacia],
                  identificador: [str, list, tuple], verify_integrity: bool = True, nombre: str = None,
                  imp: bool = True) -> BaseDatos:
    """
    Junta dos bases con identificadores distintos entre ambos
    Se valida que no haya ID's repetidos
    Importante: Se sobreescribirá la metadata de [base_aba] con la de [base_arr]
    TODO: Agregar una opción para escoger qué metadata se queda
    """
    # Chequeos de bases vacías
    if not isinstance(base_arr, BaseDatos) and not isinstance(base_aba, BaseDatos):
        raise ErrBDD("Se está intentando juntar dos bases vacías.")
    if not isinstance(base_arr, BaseDatos):
        return base_aba
    if not isinstance(base_aba, BaseDatos):
        return base_arr

    if isinstance(identificador, str):
        identificador = [identificador]
    bd_ret = juntar_metadata(base_arr, base_aba, nombre=nombre)
    # Aquí se toman los DataFrame de los objetos BaseDatos y se pone(n) la(s) variable(s) [on] como Index para
    # poder aplicar pd.concat y validar que no haya repetidos
    bd_ret.df = pd.concat([base_arr.df.set_index(identificador, verify_integrity=verify_integrity),
                           base_aba.df.set_index(identificador, verify_integrity=verify_integrity)],
                          verify_integrity=verify_integrity)
    # Se regresa el Index a columna(s) y se regresa al orden original
    bd_ret.df.reset_index(inplace=True)
    bd_ret.df = bd_ret.df.loc[:, bd_ret.vars_base]

    # Imprime las diferencias en las variables
    vars_arr_no_aba = [nv for nv in base_arr.vars_base if nv not in base_aba.vars_base]
    if vars_arr_no_aba and imp:
        print(f'\nVariables en [{base_arr.nombre}] que no están en [{base_aba.nombre}]:')
        for v in vars_arr_no_aba:
            print(f'\t{v}')

    vars_aba_no_arr = [nv for nv in base_aba.vars_base if nv not in base_arr.vars_base]
    if vars_aba_no_arr and imp:
        print(f'\nVariables en [{base_aba.nombre}] que no están en [{base_arr.nombre}]:')
        for v in vars_aba_no_arr:
            print(f'\t{v}')
    print()
    print(f'Base [{bd_ret.nombre}] creada')

    return bd_ret

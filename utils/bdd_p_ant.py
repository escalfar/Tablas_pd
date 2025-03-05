# -*- coding: UTF-8 -*-

from __future__ import print_function

import os
from collections import Counter
import pandas as pd
from savReaderWriter import SavReader, SavHeaderReader, SavWriter

try:
    # noinspection PyCompatibility
    from itertools import izip

    zip = izip
except ImportError:
    pass

try:
    # noinspection PyCompatibility
    basestring
except NameError:
    basestring = str


def rng(cad):
    """

    :param cad:
    :return:
    """
    ranges = (x.split(":") for x in cad.split(","))
    return [i for r in ranges for i in range(int(r[0]), int(r[-1]) + 1)]


class ErrBDD(Exception):
    pass


class MetaData:
    def __init__(self, ruta, utf8=True, nombre=None):
        with SavHeaderReader(ruta, ioUtf8=utf8) as header:
            self._metadata = header.dataDictionary()

        if nombre is None:
            nombre = 'mt_' + os.path.split(ruta)[1].split('.')[0]
        self.nombre = nombre

        # self.varNames = self.metadata['varNames']
        # self.varTypes = self.metadata['varTypes']
        # self.valueLabels = self.metadata['valueLabels']
        # self.varLabels = self.metadata['varLabels']
        # self.formats = self.metadata['formats']
        # self.missingValues = self.metadata['missingValues']
        # self.measureLevels = self.metadata['measureLevels']
        # self.columnWidths = self.metadata['columnWidths']
        # self.alignments = self.metadata['alignments']
        # self.varSets = self.metadata['varSets']
        # self.varRoles = self.metadata['varRoles']
        # self.varAttributes = self.metadata['varAttributes']
        # self.fileAttributes = self.metadata['fileAttributes']
        # self.fileLabel = self.metadata['fileLabel']
        # self.multRespDefs = self.metadata['multRespDefs']
        # self.caseWeightVar = self.metadata['caseWeightVar']

    def get_mtdt(self):
        return self._metadata

    def vars(self, *rangos):
        """

        :param rangos:
        :return:
        """

        vars_base = self._metadata['varNames'][:]

        if not rangos:
            return vars_base

        l_vars = []

        for rango in rangos:
            if isinstance(rango, basestring):
                if rango not in vars_base:
                    raise ErrBDD('No existe {} en la base de datos'.format(rango))
                l_vars.append(rango)
            else:
                msj_err = None
                ind_ini, ind_fin = None, None
                try:
                    ind_ini, ind_fin = tuple(vars_base.index(nvar) for nvar in rango)
                except ValueError as e:
                    msj_err = e.message
                else:
                    if ind_ini >= ind_fin:
                        msj_err = 'Error en orden de rango de variables: {}'.format(rango)

                if msj_err:
                    raise ErrBDD('Grupo mal especificado: ' + msj_err)

                l_vars.extend(vars_base[ind_ini: ind_fin + 1])

        duplicados = [item for item, count in Counter(l_vars).items() if count > 1]

        if duplicados:
            raise ErrBDD('Variables duplicadas: {}'.format(tuple(duplicados)))

        return l_vars


class BaseDatos:
    """
    Base de datos

    """

    atrib_var = [
        'valueLabels',
        'varTypes',
        'varSets',
        'varAttributes',
        'varRoles',
        'measureLevels',
        'varLabels',
        'formats',
        'columnWidths',
        'alignments',
        'missingValues',
    ]

    def __init__(self, ruta, utf8=True, raw=False, nombre=None):
        # with SavHeaderReader(ruta, ioUtf8=utf8) as header:
        #     self.metadata = header.dataDictionary()

        self.obj_metadata = MetaData(ruta, utf8, nombre)
        self.metadata = self.obj_metadata.get_mtdt()

        if nombre is None:
            nombre = os.path.split(ruta)[1].split('.')[0]
        self.nombre = nombre
        self.ruta = ruta

        with SavReader(self.ruta, ioUtf8=utf8, rawMode=raw) as data:
            self.df = pd.DataFrame(list(data), columns=self.metadata['varNames'])

        print('Base ' + self.ruta)

    def vars(self, *rangos):
        return self.obj_metadata.vars(*rangos)

    def eti(self, nombre):
        return self.metadata['varLabels'][nombre]

    def etiq_v(self, l_var, etiqueta):
        """
        :param l_var:
        :param etiqueta:
        """
        if isinstance(l_var, (str, unicode)):
            l_var = self.vars(l_var)
        else:
            l_var = self.vars(*l_var)

        if callable(etiqueta):
            callback = etiqueta
        else:
            def callback(*args):
                return etiqueta

        for ind, nvar in enumerate(l_var):
            eti_o = self.metadata['varLabels'][nvar]
            self.metadata['varLabels'][nvar] = callback(nvar, eti_o, ind)

    def etival(self, nombre):
        """
        :param nombre:
        :return:
        """
        return self.metadata['valueLabels'][nombre]

    def etival_v(self, l_var, etiquetas):
        """
        :param l_var:
        :param etiquetas:
        """
        if isinstance(l_var, (str, unicode)):
            l_var = self.vars(l_var)
        else:
            l_var = self.vars(*l_var)

        if callable(etiquetas):
            callback = etiquetas
        else:
            def callback(*args):
                return etiquetas

        for ind, nvar in enumerate(l_var):
            etis_o = self.metadata['valueLabels'][nvar]
            self.metadata['valueLabels'][nvar] = callback(nvar, etis_o, ind)

    def frm_v(self, l_var, formatos):
        """
        :param l_var:
        :param formatos:
        """
        if isinstance(l_var, (str, unicode)):
            l_var = self.vars(l_var)
        else:
            l_var = self.vars(*l_var)

        if callable(formatos):
            callback = formatos
        else:
            def callback(*args):
                return formatos

        for ind, nvar in enumerate(l_var):
            frms_o = self.metadata['formats'][nvar]
            frms_n = callback(nvar, frms_o, ind)
            self.metadata['formats'][nvar] = frms_n
            tipo_n = self.metadata['formats'][nvar]
            if 'A' in frms_n:
                tipo_n = int(frms_n.replace('A', ''))
            elif 'F' in frms_n:
                tipo_n = 0
            self.metadata['varTypes'][nvar] = tipo_n

    def renomb_v(self, l_var, nombre):
        """
        :param l_var:
        :param nombre:
        """
        if isinstance(l_var, (str, unicode)):
            l_var = self.vars(l_var)
        else:
            l_var = self.vars(*l_var)

        if callable(nombre):
            callback = nombre
        else:
            def callback(*args):
                return nombre

        for ind, nvar in enumerate(l_var):
            ind_var = self.vars().index(nvar)
            eti_o = self.metadata['varLabels'][nvar]
            nuevo_nvar = callback(nvar, eti_o, ind)

            self.metadata['varNames'][ind_var] = nuevo_nvar

            # Modificar metadatos
            for atributo in BaseDatos.atrib_var:
                if nvar in self.metadata[atributo]:
                    self.metadata[atributo][nuevo_nvar] = self.metadata[atributo].pop(nvar)

            self.df.rename(columns={nvar: nuevo_nvar}, inplace=True)

        duplicados = [elem for elem, cuenta in Counter(self.vars()).items() if cuenta > 1]

        if duplicados:
            raise ErrBDD('Variables duplicadas: {}'.format(tuple(duplicados)))

    def incl_v(self, l_var, elim=False, orden_base=False, todas=False):
        """
        Sï¿½lo deja las variables listadas
        :param l_var:
        :param elim:
        :param orden_base:
        :param todas:
        """
        if elim and todas:
            raise ErrBDD(u'Se eliminarï¿½an todas las variables de la base.')

        db_vars = self.vars()

        if isinstance(l_var, (str, unicode)):
            l_var = self.vars(l_var)
        else:
            l_var = self.vars(*l_var)

        if elim:
            l_var = [nv for nv in db_vars if nv not in l_var]

        if orden_base:
            l_var = [nv for nv in db_vars if nv in l_var]

        if todas:
            l_var += [nv for nv in db_vars if nv not in l_var]

        # Modificar metadatos
        for atributo in BaseDatos.atrib_var:
            dicc_atr = self.metadata[atributo]
            self.metadata[atributo] = {nom_var: val for nom_var, val in dicc_atr.items() if nom_var in l_var}

        self.df = self.df.loc[:, l_var]

        self.metadata['varNames'] = l_var

    def borr_v(self, l_var):
        """
        Elimina las variables listadas
        :param l_var:
        """

        self.incl_v(l_var, elim=True)

    def crear_v(self, nombre, antes=None, despues=None, medida='nominal',
                eti=None, fmt='F3', tipo=0, eti_vals=None):
        """
        Crea una nueva variable
        :param medida:
        :param tipo:
        :param nombre:
        :param antes:
        :param despues:
        :param eti:
        :param fmt:
        :param eti_vals:
        """
        l_vars = self.vars()
        if nombre in l_vars:
            raise ErrBDD('Ya existe la variable {}'.format(nombre))

        if antes and despues:
            raise ErrBDD('Sï¿½lo debe especificarse un parï¿½metro dentro de [antes, despues]')
        elif antes or despues:
            err_msj = None
            if antes is not None and antes not in l_vars:
                err_msj = antes
            if despues is not None and despues not in l_vars:
                err_msj = despues

            if err_msj:
                raise ErrBDD('No existe la variable {}'.format(err_msj))
            pos = antes or despues
        else:
            pos = l_vars[-1]

        # ï¿½ndice de inserciï¿½n
        indice = l_vars.index(pos)
        if despues or not antes:
            indice += 1

        # Agregar en datos
        self.df.insert(indice, nombre, None)

        # Agregar en metadatos
        def ag_atrib(atrib, val_atr):
            self.metadata[atrib][nombre] = val_atr

        ag_atrib('varTypes', tipo)
        ag_atrib('valueLabels', eti_vals if eti_vals is not None else {})
        ag_atrib('varRoles', 'input')
        ag_atrib('measureLevels', medida)
        ag_atrib('varLabels', eti if eti is not None else nombre)
        ag_atrib('formats', fmt)
        ag_atrib('columnWidths', 8)
        ag_atrib('alignments', 'right')
        ag_atrib('missingValues', {})

        # Agregar en variables
        l_vars.insert(indice, nombre)
        self.metadata['varNames'] = l_vars

    def comput_v(self, nombre, valor, crear=False):
        """
        Calcula un valor y lo asigna a la variable
        :param nombre:
        :param valor:
        :param crear:
        :return: None
        """
        if crear:
            self.crear_v(nombre)

        l_vars = self.vars()
        if nombre not in l_vars:
            raise ErrBDD('No existe la variable {}'.format(nombre))

        if callable(valor):
            callback = valor
        else:
            def callback(r):
                return valor

        self.df.loc[:, nombre] = self.df.apply(callback, axis=1)

    def recod(self, nombre, rec):
        """
        Recodifica una variable
        :param nombre: nombre o nombres de las variables
        :param rec: diccionario con las recodificaciones
        :return: None
        """
        if isinstance(nombre, (str, unicode)):
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

    def guardar(self, ruta, utf8=True):
        """
        Guarda la base de datos en disco
        :param ruta:
        :param utf8:
        """
        # print(self.df.loc[:, 'sexo'])
        self.df = self.df.where(self.df.notna(), None)
        with SavWriter(ruta, ioUtf8=utf8, **self.metadata) as writer:
            for row in self.df.itertuples(False, None):
                writer.writerow(list(row))

        print('Guardada en ' + ruta)

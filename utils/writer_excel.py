import os
import pandas as pd


class ErrWrtExl(Exception):
    pass


class WriterExcel:
    # TODO: Integrar BaseDatos a esta clase
    def __init__(self, ruta, nombre=None):
        print('Iniciando proceso.')
        self.writer = pd.ExcelWriter(ruta, engine='xlsxwriter')

        if nombre is None:
            nombre = os.path.split(ruta)[1].split('.')[0]
        self.nombre = nombre
        self.ruta = ruta
        self.nombres_hojas = []

    def add_tabla_a_excel(self, df: pd.DataFrame, nom_hoja: str = None) -> None:
        if nom_hoja is None:
            nom_hoja = f'_{len(self.nombres_hojas) + 1}'
        if nom_hoja in self.nombres_hojas:
            raise ErrWrtExl(f"Ya existe una hoja con nombre [{nom_hoja}] en el archivo")
        self.nombres_hojas.append(nom_hoja)
        df.to_excel(self.writer, sheet_name=nom_hoja)
        print(f'Hoja [{nom_hoja}] procesada')

    def guardar_excel(self) -> None:
        self.writer.save()
        print('Archivo guardado en ' + self.ruta)

    # def __exit__(self, exc_type, exc_val, exc_tb):
    #     print('Ahhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh...')
    #     self.guardar_excel()

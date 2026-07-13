# BlockModel Studio - paquete Streamlit

Este paquete contiene únicamente la aplicación Streamlit y sus recursos de
ejecución. No requiere Docker, Azure ni componentes SPFx.

## Instalación

Desde PowerShell, dentro de la carpeta de la aplicación:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Ejecución

```powershell
python -m streamlit run app.py --global.developmentMode=false
```

Abrir `http://localhost:8501` si el navegador no se abre automáticamente.

## Reemplazo de otra instalación

1. Respaldar la carpeta existente.
2. Conservar fuera del reemplazo cualquier modelo o reporte propio del usuario.
3. Copiar el contenido de este paquete sobre la carpeta de la otra aplicación.
4. Volver a instalar `requirements.txt` para incorporar dependencias nuevas.
5. Iniciar Streamlit y comprobar Home, carga de modelos y exportación PDF/Excel.

Los modelos de bloque no vienen incluidos. Cada usuario carga sus propios
archivos CSV, TXT o Excel durante la sesión.

from setuptools import setup, find_packages

setup(
    name="easy-mldeploy",
    version="0.1",
    description="A tool for easy ML deployment.",
    author="Bartłomiej Kiczek (barthemius)",
    author_email="bartlomiej.kiczek@gmail.com",
    license="MIT",
    packages=find_packages(),
    include_package_data=True,
    install_requires=["fastapi", "uvicorn", "numpy", "pydantic"],
    zip_safe=False,
)

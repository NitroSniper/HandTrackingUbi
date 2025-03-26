# mediapipe_pkg.nix
{
  lib,
  buildPythonPackage,
  fetchPypi,
  setuptools,
  wheel,
  python,
  fetchurl,
}:

buildPythonPackage rec {
  pname = "mediapipe";
  version = "0.10.21";
  format = "wheel";

  pyShortVersion = "cp${builtins.replaceStrings [ "." ] [ "" ] python.pythonVersion}";
  src = fetchurl {
    url = "https://files.pythonhosted.org/packages/9f/99/5da7ae7f7e25847383bc2fe5a9adc7ce150dd371682f486c0666b407cad7/mediapipe-0.10.21-cp312-cp312-manylinux_2_28_x86_64.whl";
    hash = "sha256-lW6x68J1xinmGwhbLKuJw6W56TutG7EHNI2Y2vtaS7U=";
  };
  # do not run tests
  doCheck = false;
  build-system = [
    setuptools
    wheel
  ];
}

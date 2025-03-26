# shell.nix
let
  pkgs = import <nixpkgs> { };

  python = pkgs.python312.override {
    self = python;
    packageOverrides = pyfinal: pyprev: {
      mediapipe = pyfinal.callPackage ./mediapipe_pkg.nix { };
    };
  };

in
pkgs.mkShell {
  shellHook = ''
    # fixes libstdc++ issues and libgl.so issues
    LD_LIBRARY_PATH=${pkgs.stdenv.cc.cc.lib}/lib/:/run/opengl-driver/lib/
  '';
  packages = [
    (python.withPackages (python-pkgs: [
      # select Python packages here
      python-pkgs.mediapipe
    ]))
  ];
}

{
  description = "pplx-agent-tools — agent toolkit for Perplexity, backed by your Pro subscription";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    uv2nix = {
      url = "github:pyproject-nix/uv2nix";
      inputs = {
        pyproject-nix.follows = "pyproject-nix";
        nixpkgs.follows = "nixpkgs";
      };
    };
    pyproject-build-systems = {
      url = "github:pyproject-nix/build-system-pkgs";
      inputs = {
        pyproject-nix.follows = "pyproject-nix";
        uv2nix.follows = "uv2nix";
        nixpkgs.follows = "nixpkgs";
      };
    };
  };

  outputs =
    {
      self,
      nixpkgs,
      pyproject-nix,
      uv2nix,
      pyproject-build-systems,
      ...
    }:
    let
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "aarch64-darwin"
        "x86_64-darwin"
      ];
      forAllSystems = nixpkgs.lib.genAttrs systems;

      # Version derived from .git_archival.txt at flake-eval time. The file is
      # populated by GitHub when serving the tarball for a tagged ref (via
      # `export-subst` in .gitattributes) and contains `describe-name: vX.Y.Z`.
      # Without this, uv2nix sees `editable = "."` in uv.lock with no version
      # field and hatch-vcs/setuptools-scm has no git history → falls back to
      # "0.0.0". We extract the describe-name and pass it as
      # SETUPTOOLS_SCM_PRETEND_VERSION so hatch-vcs uses it.
      #
      # Returns null when the substitution didn't run (local working-tree builds
      # without a real tag), so consumers know to fall back.
      versionFromArchival =
        let
          path = ./.git_archival.txt;
          contents = if builtins.pathExists path then builtins.readFile path else "";
          match = builtins.match ".*describe-name: (v?[0-9][^\n]*).*" contents;
        in
        if match == null then null else nixpkgs.lib.removePrefix "v" (builtins.head match);

      # Native overrides for wheels that ship bundled shared libs not on a
      # NixOS-style linker path. macOS wheels are relative-linked and need
      # nothing; Linux manylinux wheels need autoPatchelfHook + the libs
      # their bundled .so's dlopen.
      #
      # curl_cffi:    bundles libcurl-impersonate-chrome.so → needs zlib + openssl
      # onnxruntime:  bundles libonnxruntime.so → needs libstdc++ (via cc.cc.lib)
      mkPyprojectOverrides =
        pkgs: _final: prev:
        nixpkgs.lib.optionalAttrs pkgs.stdenv.hostPlatform.isLinux {
          curl-cffi = prev.curl-cffi.overrideAttrs (old: {
            nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [ pkgs.autoPatchelfHook ];
            buildInputs = (old.buildInputs or [ ]) ++ [
              pkgs.zlib
              pkgs.openssl
            ];
          });
          onnxruntime = prev.onnxruntime.overrideAttrs (old: {
            nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [ pkgs.autoPatchelfHook ];
            buildInputs = (old.buildInputs or [ ]) ++ [ pkgs.stdenv.cc.cc.lib ];
          });
        }
        // nixpkgs.lib.optionalAttrs (versionFromArchival != null) {
          # When we know the version (tagged release fetched as a tarball),
          # force hatch-vcs to use it instead of falling back to 0.0.0.
          pplx-agent-tools = prev.pplx-agent-tools.overrideAttrs (old: {
            env = (old.env or { }) // {
              SETUPTOOLS_SCM_PRETEND_VERSION = versionFromArchival;
            };
          });
        };

      mkPythonSet =
        system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
          workspace = uv2nix.lib.workspace.loadWorkspace { workspaceRoot = ./.; };
          # sourcePreference = "wheel" — curl_cffi, onnxruntime, rookiepy,
          # tokenizers, lxml all ship usable wheels; building from sdist
          # would require their full native toolchains in nixpkgs.
          overlay = workspace.mkPyprojectOverlay { sourcePreference = "wheel"; };
        in
        {
          inherit workspace;
          pythonSet =
            (pkgs.callPackage pyproject-nix.build.packages {
              python = pkgs.python312;
            }).overrideScope
              (
                nixpkgs.lib.composeManyExtensions [
                  pyproject-build-systems.overlays.wheel
                  overlay
                  (mkPyprojectOverrides pkgs)
                ]
              );
        };

      # Bundle the runtime venv with SKILL.md at the ak2k-skills convention
      # path ($out/share/skills/<name>/) so ak2k-skills' registry can pick
      # it up the same way as krisp-cli / claude-sessions / msgvault-query.
      # SKILL.md is also force-included in the wheel under pplx_agent_tools/
      # for non-Nix consumers (`pplx skill-path`).
      mkPplx =
        system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
          s = mkPythonSet system;
          venv = s.pythonSet.mkVirtualEnv "pplx-agent-tools-env" s.workspace.deps.default;
        in
        pkgs.symlinkJoin {
          name = "pplx-agent-tools";
          paths = [ venv ];
          postBuild = ''
            mkdir -p $out/share/skills/pplx-agent-tools
            cp ${./SKILL.md} $out/share/skills/pplx-agent-tools/SKILL.md
          '';
          meta = {
            description = "Agent toolkit for Perplexity, backed by your Pro subscription's web session cookies";
            homepage = "https://github.com/ak2k/pplx-agent-tools";
            license = nixpkgs.lib.licenses.mit;
            mainProgram = "pplx";
            platforms = systems;
          };
        };
    in
    {
      packages = forAllSystems (
        system:
        let
          s = mkPythonSet system;
        in
        {
          default = mkPplx system;
          pplx-agent-tools = mkPplx system;
          # Dev venv — includes pytest, ruff, basedpyright, vulture. Used by the
          # checks and devShell below; rarely useful as a `nix build` target.
          dev = s.pythonSet.mkVirtualEnv "pplx-agent-tools-env-dev" s.workspace.deps.all;
        }
      );

      checks = forAllSystems (
        system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
          s = mkPythonSet system;
          devEnv = s.pythonSet.mkVirtualEnv "pplx-agent-tools-env-dev" s.workspace.deps.all;
        in
        {
          # Wraps `nix build .#default` as a check so `nix flake check`
          # surfaces wheel/native-build regressions on every CI run.
          package = self.packages.${system}.default;

          tests = pkgs.runCommand "pplx-agent-tools-tests" { nativeBuildInputs = [ devEnv ]; } ''
            cp -r ${./.}/. .
            chmod -R u+w .
            export HOME=$TMPDIR
            export PYTEST_CACHE_DIR=$TMPDIR/.pytest_cache
            pytest -q
            touch $out
          '';

          lint = pkgs.runCommand "pplx-agent-tools-lint" { nativeBuildInputs = [ devEnv ]; } ''
            cp -r ${./.}/. .
            chmod -R u+w .
            export RUFF_CACHE_DIR=$TMPDIR/.ruff_cache
            ruff check .
            ruff format --check .
            touch $out
          '';

          # Use nixpkgs' basedpyright (which bundles its own node), not
          # the Python `basedpyright` wrapper from the dev venv — that
          # wrapper downloads node via nodeenv on first run, which fails
          # in the Nix build sandbox (no network).
          typecheck =
            pkgs.runCommand "pplx-agent-tools-typecheck"
              {
                nativeBuildInputs = [
                  pkgs.basedpyright
                  devEnv
                ];
              }
              ''
                cp -r ${./.}/. .
                chmod -R u+w .
                export HOME=$TMPDIR
                # Point basedpyright at the dev venv so third-party imports resolve.
                export PYTHONPATH="${devEnv}/lib/python3.12/site-packages"
                basedpyright --pythonpath ${devEnv}/bin/python pplx_agent_tools
                touch $out
              '';

          deadcode = pkgs.runCommand "pplx-agent-tools-deadcode" { nativeBuildInputs = [ devEnv ]; } ''
            cp -r ${./.}/. .
            chmod -R u+w .
            vulture
            touch $out
          '';

          nix-fmt = pkgs.runCommand "pplx-agent-tools-nix-fmt" { nativeBuildInputs = [ pkgs.nixfmt ]; } ''
            nixfmt --check ${./flake.nix}
            touch $out
          '';
        }
      );

      devShells = forAllSystems (
        system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
          s = mkPythonSet system;
          devEnv = s.pythonSet.mkVirtualEnv "pplx-agent-tools-env-dev" s.workspace.deps.all;
        in
        {
          default = pkgs.mkShell {
            packages = [
              devEnv
              pkgs.uv
            ];
            shellHook = ''
              # uv should use the venv we built, not its own python-build-standalone
              export UV_NO_SYNC=1
              export UV_PYTHON="${devEnv}/bin/python"
            '';
          };
        }
      );
    };
}

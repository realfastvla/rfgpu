#! /bin/bash

VCUB="1.8.0"
[ -d cub-$VCUB ] || ( wget -O - https://github.com/NVlabs/cub/archive/v$VCUB.tar.gz | tar xzf - )

VPYBIND="2.2.4"
[ -d pybind11-$VPYBIND ] || ( wget -O - https://github.com/pybind/pybind11/archive/v$VPYBIND.tar.gz | tar xzf - )

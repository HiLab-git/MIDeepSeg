import sys
import numpy as np
from distutils.core import setup
from distutils.extension import Extension

package_name = 'crf_max_flow'
module_name1 = 'max_flow'
module_name2 = 'max_flow3d'

v = sys.version[0]
source_2d = 'py{0:}/max_flow.cpp'.format(v)
source_3d = 'py{0:}/max_flow3d.cpp'.format(v)

module1 = Extension(module_name1,
                    include_dirs = [np.get_include()],
                    sources = [source_2d, 'maxflow-v3.0/graph.cpp', 'maxflow-v3.0/maxflow.cpp'])
module2 = Extension(module_name2,
                    include_dirs = [np.get_include()],
                    sources = [source_3d, 'maxflow-v3.0/graph.cpp', 'maxflow-v3.0/maxflow.cpp'])
setup(name=package_name,
      ext_modules = [module1, module2])

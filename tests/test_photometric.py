# Copyright 2025 Artezaru
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy
import pytest

import pysdic

@pytest.mark.parametrize("Np", [0, 1, 5])
@pytest.mark.parametrize("Nl", [0, 1, 3])
def test_bouguer_law_shape(Np, Nl):
    numpy.random.seed(0)
    
    if Np != 0:
        surface_points = numpy.random.rand(Np, 3)
        surface_normals = numpy.repeat([[0, 0, 1]], Np, axis=0) # Upward normals
    else:
        surface_points = numpy.random.rand(3) # Shape: (3,)
        surface_normals = numpy.array([0, 0, 1]) # Upward normal
        Np = 1 # Treat as single point
        
    if Nl != 0:
        light_sources = numpy.random.rand(Nl, 3) 
        light_sources[:, 2] += 5.0 # Up along Z
    else:
        light_sources = numpy.random.rand(3) # Shape: (3,)
        light_sources[2] += 5.0 # Up along Z
        Nl = 1 # Treat as single light source
        
    intensities = pysdic.compute_bouguer_law(surface_points, surface_normals, light_sources)
    assert intensities.shape == (Np, Nl)
    
    
@pytest.mark.parametrize("Np", [0, 1, 5])
@pytest.mark.parametrize("Nl", [0, 1, 3])
@pytest.mark.parametrize("No", [0, 1, 2])
@pytest.mark.parametrize("Nw", [0, 1, 4])
def test_brdf_ward(Np, Nl, No, Nw): 
    numpy.random.seed(0) 
    
    if Np != 0: 
        surface_points = numpy.random.rand(Np, 3)
        surface_normals = numpy.repeat([[0, 0, 1]], Np, axis=0)
        assert surface_normals.shape == (Np, 3)
    else:
        surface_points = numpy.random.rand(3)
        surface_normals = numpy.array([0, 0, 1])
        Np = 1 # Treat as single point
                
    if Nl != 0: 
        light_sources = numpy.random.rand(Nl, 3)
        light_sources[:, 2] += 5.0
        assert light_sources.shape == (Nl, 3)
    else:
        light_sources = numpy.random.rand(3)
        light_sources[2] += 5.0
        Nl = 1
        
    if No != 0: 
        observers = numpy.random.rand(No, 3)
        assert observers.shape == (No, 3)
        observers[:, 2] += 5.0
    else:
        observers = numpy.random.rand(3)
        observers[2] += 5.0
        No = 1

    if Nw != 0:
        parameters = numpy.repeat([[0.1, 0.1, 0.5]], Nw, axis=0)
        assert parameters.shape == (Nw, 3)
    else:
        parameters = numpy.array([0.1, 0.1, 0.5])
        Nw = 1

    brdf = pysdic.compute_brdf_ward(
        surface_points,
        surface_normals,
        light_sources,
        observers,
        parameters
    )
    assert brdf.shape == (Np, Nl, No, Nw)




@pytest.mark.parametrize("Np", [0, 1, 5])
@pytest.mark.parametrize("Nl", [0, 1, 3])
@pytest.mark.parametrize("No", [0, 1, 2])
@pytest.mark.parametrize("Nw", [0, 1, 4])
def test_brdf_beckmann(Np, Nl, No, Nw): 
    numpy.random.seed(0) 
    
    if Np != 0: 
        surface_points = numpy.random.rand(Np, 3)
        surface_normals = numpy.repeat([[0, 0, 1]], Np, axis=0)
        assert surface_normals.shape == (Np, 3)
    else:
        surface_points = numpy.random.rand(3)
        surface_normals = numpy.array([0, 0, 1])
        Np = 1 # Treat as single point
                
    if Nl != 0: 
        light_sources = numpy.random.rand(Nl, 3)
        light_sources[:, 2] += 5.0
        assert light_sources.shape == (Nl, 3)
    else:
        light_sources = numpy.random.rand(3)
        light_sources[2] += 5.0
        Nl = 1
        
    if No != 0: 
        observers = numpy.random.rand(No, 3)
        assert observers.shape == (No, 3)
        observers[:, 2] += 5.0
    else:
        observers = numpy.random.rand(3)
        observers[2] += 5.0
        No = 1

    if Nw != 0:
        parameters = numpy.repeat([[0.1, 0.1, 0.5]], Nw, axis=0)
        assert parameters.shape == (Nw, 3)
    else:
        parameters = numpy.array([0.1, 0.1, 0.5])
        Nw = 1

    brdf = pysdic.compute_brdf_beckmann(
        surface_points,
        surface_normals,
        light_sources,
        observers,
        parameters
    )
    assert brdf.shape == (Np, Nl, No, Nw)
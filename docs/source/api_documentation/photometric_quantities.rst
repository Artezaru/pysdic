Photometric Quantities
=============================

.. contents:: Table of Contents
   :local:
   :depth: 2
   :backlinks: top


Irradiance (:math:`E`)
------------------------

The irradiance at a point on a surface is defined as the power of incoming light per unit area. It is measured in Watts per square meter (:math:`W/m^2`).
The irradiance is noted as :math:`E` and is defined mathematically as:

.. math::

    E(x, y) = \frac{dF(x, y)}{dS}

where :math:`F(x, y)` is the radiant flux (W) incident on the surface at point :math:`(x, y)`, and :math:`S` is the area of the surface.

The irradiance is called "illuminance" when considering visible light, and is measured in lumens per square meter (lux).

.. figure:: /_static/photometric/irradiance.png
   :alt: Irradiance 
   :align: center
   :width: 250px


Intensity of a Light Source (:math:`I`)
----------------------------------------

The intensity of a light source is the flux per unit of solid angle in a given observation direction. 
It is measured in Watts per steradian (:math:`W/sr`) or in lumens per steradian (candela) for visible light.
The intensity is noted as :math:`I` and is defined mathematically as:

.. math::

    I(\theta, \phi) = \frac{dF(\theta, \phi)}{d\Omega}

where :math:`F(\theta, \phi)` is the radiant flux (W) emitted by the light source in the direction defined by the angles :math:`(\theta, \phi)`, and :math:`\Omega` is the solid angle in steradians.

.. figure:: /_static/photometric/intensity.png
   :alt: Intensity
   :align: center
   :width: 250px


Radiance (:math:`L`)
----------------------

Lets consider an elementary surface area :math:`dS` of the source emitting an intensity :math:`dI(\theta, \phi)` in the direction defined by the angles :math:`(\theta, \phi)`.
The radiance at point :math:`(x, y)` in the direction :math:`(\theta, \phi)` is defined as the flux per unit projected area and per unit solid angle.
It is measured in Watts per steradian per square meter (:math:`W/sr/m^2`) or in candelas per square meter for visible light.
The radiance is noted as :math:`L` and is defined mathematically as:

.. math::

    L(x, y, \theta, \phi) = \frac{dI(\theta, \phi)}{dS \cdot \cos(\theta_S)} = \frac{d^2F(\theta, \phi)}{dS \cdot \cos(\theta_S) \cdot d\Omega} = n^2 \cdot \frac{d^2F}{d^2G}
     
where :math:`\theta_S` is the angle between the normal to the surface area :math:`dS` and the direction :math:`(\theta, \phi)` of the emitted light, :math:`n` is the refractive index of the medium surrounding the source, and :math:`G` is the geometric extent.

.. figure:: /_static/photometric/radiance.png
   :alt: Radiance
   :align: center
   :width: 250px

For a emitting inside a finite cone in air with uniform radiance :math:`L_0`, the radiance is given by:

.. math::

    L_0 = \frac{F_{tot}}{\pi \cdot S \cdot \sin^2(\theta)}

where :math:`F_{tot}` is the total flux emitted by the source, :math:`S` is the area of the emitting surface, and :math:`\theta` is the half-angle of the emission cone.

.. figure:: /_static/photometric/radiance_cone.png
   :alt: Radiance Cone
   :align: center
   :width: 250px

A lambertian source is a surface that emits light uniformly in all directions.
In this case, the radiance is constant and does not depend on the direction :math:`(\theta, \phi)`.

.. math::

    L_{lambertian} = \frac{F_{tot}}{\pi \cdot S}


Bouguer Law for Irradiance
----------------------------

For a source of intensity :math:`I` located at a distance :math:`d` from a surface, the irradiance :math:`E` on the surface is given by Bouguer's law:

.. math::

    E = \frac{I}{d^2} \cdot \cos(\theta)

where :math:`\theta` is the angle between the normal to the surface and the direction of the incoming light from the source.

.. figure:: /_static/photometric/bouguer_law.png
   :alt: Bouguer Law
   :align: center
   :width: 250px


Geometric extent (:math:`G`)
--------------------------------

The geometric extent between two surfaces quantifies the ability of light to transfer from one surface to another.
Is is a conserved quantity in an optical system without losses.

The geometric extent between two surfaces :math:`S_1` and :math:`S_2` is expressed in square meters  per steradian (:math:`m^2/sr`) and is noted as :math:`G` :

.. math::

    G = \frac{dS_1 \cdot dS_2 \cdot \cos(\theta_1) \cdot \cos(\theta_2)}{d^2}

where :math:`dS_1` and :math:`dS_2` are the areas of the two surfaces, :math:`\theta_1` and :math:`\theta_2` are the angles between the normals to the surfaces and the line connecting them, and :math:`d` is the distance between the two surfaces.

.. figure:: /_static/photometric/geometric_extent.png
   :alt: Geometric Extent
   :align: center
   :width: 400px


Bidirectional Reflectance Distribution Function (BRDF)
-------------------------------------------------------

For a given surface, the BRDF is defined as the ratio of the radiance in the direction :math:`(\theta_o, \phi_o)` to the irradiance from direction :math:`(\theta_i, \phi_i)`.
The BRDF is expressed in inverse steradians (:math:`sr^{-1}`) and is noted as :math:`BRDF`.

.. math::

    BRDF(\theta_i, \phi_i, \theta_o, \phi_o) = \frac{dL_o(\theta_o, \phi_o)}{dE_i(\theta_i, \phi_i)}

where :math:`L_o` is the outgoing radiance in Watts per steradian per square meter and :math:`E_i` is the incoming irradiance in Watts per square meter.

.. figure:: /_static/photometric/brdf.png
   :alt: BRDF
   :align: center
   :width: 500px 

The amount of the incident flux emmitted in the Fresnel direction is called specular reflection, while the rest of the flux is called diffuse reflection or scattering reflection.


Optical system detection (Case of imaging system)
----------------------------------------------------

An optical imaging system collects light from a surface and forms an image on a detector.
The case of an imaging system is when the apparent size of tthe observed surface is much larger than the size of the detector's pixel.

The amount of flux :math:`F_{det}` collected by a pixel is given by:

.. math::

    F_{det} = T_{op} \cdot L_{obj} \cdot G

where :math:`T_{op}` is the optical transmission of the system, :math:`L_{obj}` is the radiance of the observed surface, and :math:`G` is the geometric extent between the observed surface and the detector pixel.
As the geometrical extent is conserved in an optical system without losses, it can be calculated at any plane in the system.

For an in-axis optical system with a circular aperture of diameter :math:`D`, the flux collected by a pixel of area :math:`S_{pix}` is given by:

.. math::

    F_{det} = T_{op} \cdot L_{obj} \cdot \frac{\pi \cdot S_{pix}}{4 N^2}

where :math:`N` is the aperture number of the system.

For an off-axis optical system, the irradiance collected by a pixel is, if the radiance is uniform in the object space :

.. math::

    E(\theta) = E(0) \cdot \cos^4(\theta)



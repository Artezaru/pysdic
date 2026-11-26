pysdic.GrayLevelGroups
======================

.. currentmodule:: pysdic

The :class:`pysdic.GrayLevelGroups` stores the groups of integration points used to compute the statistics
(mean and standard deviation) of the gray level corrections. It is created by :func:`pysdic.compute_gray_level_groups`.

- ``point_groups``: the group of each point :math:`(N_p,)`, whose statistics correct its gray level.
- ``members``: a sparse boolean matrix :math:`(N_g, N_p)`, the row :math:`k` giving the points used to compute the
  statistics of the group :math:`k`. The groups can overlap (neighborhoods of the elements).

.. autoclass:: GrayLevelGroups
   :members:
   :show-inheritance:

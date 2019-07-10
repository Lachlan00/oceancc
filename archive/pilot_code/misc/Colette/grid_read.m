function grid = grid_read( file, his_file )

% grid = grid_read( file, [his_file] )
%
% This function loads the given ROMS grid file. If a history file is given,
% then various parameters are loaded from the history file to add attributes
% to the grid.
%
%
% Created by Brian Powell on 2007-10-16.
% Copyright (c)  powellb. All rights reserved.
%

% try
  grid.filename = file;
  grid.spherical = false;
  if (nc_isvar(file,'spherical'))
    if ( nc_varget(file,'spherical') == 'T' )
      grid.spherical = true;
    end
  end
  grid.maskr = nc_varget(file, 'mask_rho');
  grid.masku = nc_varget(file, 'mask_u');
  grid.maskv = nc_varget(file, 'mask_v');
  grid.maskp = nc_varget(file, 'mask_psi');
  grid.angle = nc_varget(file, 'angle');
  if ( grid.spherical | nc_isvar(file,'lat_rho') )
    grid.latr  = nc_varget(file, 'lat_rho');
    grid.lonr  = nc_varget(file, 'lon_rho');
    grid.latu  = nc_varget(file, 'lat_u');
    grid.lonu  = nc_varget(file, 'lon_u');
    grid.latv  = nc_varget(file, 'lat_v');
    grid.lonv  = nc_varget(file, 'lon_v');
    grid.latp  = nc_varget(file, 'lat_psi');
    grid.lonp  = nc_varget(file, 'lon_psi');
  end
  if (nc_isvar(file,'x_rho'))
    grid.xr    = nc_varget(file, 'x_rho');
    grid.yr    = nc_varget(file, 'y_rho');
    grid.xu    = nc_varget(file, 'x_u');
    grid.yu    = nc_varget(file, 'y_u');
    grid.xv    = nc_varget(file, 'x_v');
    grid.yv    = nc_varget(file, 'y_v');
  end
  if (nc_isvar(file,'x_p'))
    grid.xp    = nc_varget(file, 'x_p');
    grid.yp    = nc_varget(file, 'y_p');
  end
  grid.pm    = nc_varget(file, 'pm');
  grid.pn    = nc_varget(file, 'pn');
  grid.h     = nc_varget(file, 'h');
  [grid.mp,grid.lp] = size(grid.h);
  grid.l     = grid.lp-1;
  grid.m     = grid.mp-1;
  grid.lm    = grid.lp-2;
  grid.mm    = grid.mp-2;
  grid.tracer= 2;
  
  % Grab the meta data if it exists
  if (nc_isvar(file,'N'))
    grid.n   = nc_varget(file, 'N');
  end 
  if (nc_isvar(file,'theta_s'))
    grid.theta_s = nc_varget(file, 'theta_s');
  end 
  if (nc_isvar(file,'theta_b'))
    grid.theta_b = nc_varget(file, 'theta_b');
  end 
  if (nc_isvar(file,'Tcline'))
    grid.tcline = nc_varget(file, 'Tcline');
  end 
  if (nc_isvar(file,'hc'))
    grid.hc = nc_varget(file, 'hc');
  end 
  if (nc_isvar(file,'Vtransform'))
    grid.vtransform = nc_varget(file, 'Vtransform');
  end 
  if (nc_isvar(file,'Vstretching'))
    grid.vstretching = nc_varget(file, 'Vstretching');
  end 

  % If there is a history file
  if ( nargin > 1 )
    dim = nc_getdiminfo(his_file,'N');
    grid.n = dim.Length;
    grid.theta_s = nc_varget(his_file,'theta_s');
    grid.theta_b = nc_varget(his_file,'theta_b');
    grid.tcline  = nc_varget(his_file,'Tcline');
    grid.hc      = nc_varget(his_file,'hc');
    grid.vtransform  = nc_varget(his_file,'Vtransform');
    grid.vstretching = nc_varget(his_file,'Vstretching');
  end
% catch
%   disp(lasterr);
%   error(['Problem reading: ' file]);
% end

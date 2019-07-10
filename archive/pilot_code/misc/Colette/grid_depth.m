function  [z_r,z_w,Hz] = grid_depth(grd,type,zeta);

% FUNCTION [z_r,z_w,Hz] = grid_depth(grd,type,zeta);
%
% Returns depths of coordinates at rho, u, or v points
% and at W velocity points in the vertical. Hz is the
% thickness of the sigma layer in meters.
% 
% The code is modified from E. Di Lorenzo`s rnt_setdepth

if nargin < 2 
   type='r';
end

h=grd.h;
switch type
  case 'u'
    h = h(:,2:end);
  case 'v'
    h = h(2:end,:);
end

if ~isfield(grd,'vtransform')
  grd.vtransform=1;
end
if ~isfield(grd,'vstretching')
  grd.vstretching=1;
end

eval(sprintf('mask=grd.mask%c;',type));
if nargin < 3
  zeta=zeros(size(mask));
end
N=grd.n;
   
[x,y]=size(zeta);
z_r=zeros([grd.n size(mask)]);
z_w=zeros([grd.n+1 size(mask)]);
Hz=zeros([grd.n size(mask)]);

ds=1.0/grd.n;
ods=1/ds;
tmp=mask.*h;
tmp=tmp(~isnan(tmp));
hmin=min(min(tmp));
phmax=max(max(tmp));
if isfield(grd,'hc')
	hc=grd.hc;
elseif isfield(grd,'tcline')
  hc=grd.tcline;
else
  hc=hmin;
end

theta_s=grd.theta_s;
theta_b=grd.theta_b;

% Set the stretching values
cff1=1./sinh(theta_s);
cff2=0.5/tanh(0.5*theta_s);
sc_w0=-1.0;
Cs_w0=-1.0;

if grd.vstretching==5   % Icluded by Joao Souza 05/11/2012
    for k=N:-1:1
        sc_w(k)=-(k.^2 - 2*k*N+k+N^2-N)./(N^2-N)-0.01*(k.^2-k*N)./(1-N);
    end
    for k=1:N;
        k=k-.5;
        sc_r(k+.5)=-(k.^2 - 2*k*N+k+N^2-N)./(N^2-N)-0.01*(k.^2-k*N)./(1-N);
    end 
end

for k=1:N    
    % S-coordinate stretching curves at RHO-points (C_r) and  at W-points (C_w)
    % S-coordinate at RHO-points (sc_r) and at W-points (sc_w)
    if grd.vstretching==1
      sc_w(k)=ds*(k-N);
      Cs_w(k)=(1.-theta_b)*cff1*sinh(theta_s*sc_w(k)) +theta_b*(cff2*tanh(theta_s*(sc_w(k)+0.5))-0.5);
    
      sc_r(k)=ds*((k-N)-0.5);
      Cs_r(k)=(1.-theta_b)*cff1*sinh(theta_s*sc_r(k))  +theta_b*(cff2*tanh(theta_s*(sc_r(k)+0.5))-0.5);
    elseif grd.vstretching==2
      alpha=1;
      beta=1;
      sc_w(k)=ds*(k-N);
      Csur=(1-cosh(theta_s.*sc_w(k)))/(cosh(theta_s)-1);
      Cbot=sinh(theta_b*(sc_w(k)+1))/sinh(theta_b)-1;
      weight=(sc_w(k)+1.0).^alpha.*(1.0+(alpha/beta).*(1.0-(sc_w(k)+1.0).^beta));
      Cs_w(k)=weight.*Csur+(1.0-weight).*Cbot;

      sc_r(k)=ds*((k-N)-0.5);
      Csur=(1-cosh(theta_s.*sc_r(k)))/(cosh(theta_s)-1);
      Cbot=sinh(theta_b*(sc_r(k)+1))/sinh(theta_b)-1;
      weight=(sc_r(k)+1.0).^alpha.*(1.0+(alpha/beta).*(1.0-(sc_r(k)+1.0).^beta));
      Cs_r(k)=weight.*Csur+(1.0-weight).*Cbot;
    elseif grd.vstretching==4 
      alpha=1;
      beta=1;
      sc_w(k)=ds*(k-N);
      Csur=(1-cosh(theta_s.*sc_w(k)))/(cosh(theta_s)-1);
      Cs_w(k)=(exp(theta_b*(Csur+1.0))-1)/(exp(theta_b)-1)-1;

      sc_r(k)=ds*((k-N)-0.5);
      Csur=(1-cosh(theta_s.*sc_r(k)))/(cosh(theta_s)-1);
      Cs_r(k)=(exp(theta_b*(Csur+1.0))-1)/(exp(theta_b)-1)-1;
    elseif grd.vstretching==5 % Included by Joao Souza 05/11/2012
      alpha=1;
      beta=1;
      Csur=(1-cosh(theta_s.*sc_w(k)))/(cosh(theta_s)-1);
      Cs_w(k)=(exp(theta_b*(Csur+1.0))-1)/(exp(theta_b)-1)-1;
      
      Csur=(1-cosh(theta_s.*sc_r(k)))/(cosh(theta_s)-1);
      Cs_r(k)=(exp(theta_b*(Csur+1.0))-1)/(exp(theta_b)-1)-1;
    end
end

% Set the transform values
z_w(1,:,:)=-h;
hinv=1./h;
for k=1:N
  if grd.vtransform==1
    cff_w=hc*(sc_w(k)-Cs_w(k));
    cff1_w=Cs_w(k);
    cff2_w=sc_w(k)+1.;
    
    cff_r=hc*(sc_r(k)-Cs_r(k));
    cff1_r=Cs_r(k);
    cff2_r=sc_r(k)+1.;
    
    % Depth of sigma coordinate at W-points
    z_w0=cff_w+cff1_w*h;
    z_w(k+1,:,:)=z_w0+zeta(:,:).*(1.+z_w0.*hinv);
    
    % Depth of sigma coordinate at RHO-points
    z_r0=cff_r+cff1_r*h;
    z_r(k,:,:)=z_r0+zeta(:,:).*(1.+z_r0.*hinv);
  elseif grd.vtransform == 2
    % Depth of sigma coordinate at W-points
    cff = 1./(hc + h);
    cff_w = hc*sc_w(k) + h*Cs_w(k);
    z_w(k+1,:,:)=zeta + ( zeta + h).*cff_w.*cff;

    % Depth of sigma coordinate at RHO-points
    cff_r = hc*sc_r(k) + h*Cs_r(k);
    z_r(k,:,:)=zeta + ( zeta + h ).*cff_r.*cff;
  end
end

%
% Hz
%
z_w(end,:,:)=zeta;
k=2:grd.n+1;
Hz=(z_w(k,:,:)-z_w(k-1,:,:));
%if opt ==1
%    Hz=ods*(z_w(:,:,k,t)-z_w(:,:,k-1,t));
%end

function [] = soft_threshold_proto()
  rng('default');
  x = ones(10000, 1) - 2*rand(10000, 1);
  lambda = .5;
  A = rand(100, 10000);


  % sgn(xi)*(abs(xi) - lambda)_+
  tic
  for i = 1:10000
    xproj1 = bsxfun(@times, sign(x), max((abs(x) - lambda), 0));
  end
  toc
  
  
  % THIS ONE IS MUCH FASTER!?!
  tic
  for i = 1:10000
    xproj2 = max(x - lambda, 0) + min(x + lambda, 0);
    % xproj2 = abs(xproj2);
  end
  
  toc
  norm(xproj1-xproj2)
  tic;

  
  
%{  
  for i = 1:1000
    A*xproj;
  end
  toc
  
  xproj = sparse(xproj);
  tic;
  for i = 1:1000
    A*xproj;
  end
  toc
%}
  
end
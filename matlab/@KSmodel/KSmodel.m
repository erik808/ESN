classdef KSmodel < handle
% KSmodel: Kuramoto-Sivashinsky PDE, normalized to the interval
% [0,1] with periodic boundaries.
% d/dt u + u du/dx + (1+epsilon)*(d/dx)^2 u + nu * (d/dx)^4 u = 0

    properties
        % str identifier for class
        name = 'KSmodel';

        % domain length parameter
        L  = 100

        % total number of grid points
        N  = 200

        % grid points in the x-direction
        nx

        % grid points in the y-direction
        ny  = 1; % not relevant but should be available

        % number of unknowns
        nun = 1 ;% not relevant but should be available

        % grid increment size
        dx

        % damping parameter #TODO
        nu = 1

        % initial state
        x_init

        % first derivative, central discretization
        D1

        % second derivative, central discretization
        D2

        % fourth derivative, central discretization
        D4

        % linear part of Jacobian
        Jlin

        % perturbation of the positive diffusion
        epsilon = 0.0

        % Newton tolerance
        Ntol = 1e-6;

        % Max Newton iterations
        Nkmx = 10;

        % Optional modes
        V

        initialized = false;

    end

    methods
        function self = KSmodel(L, N)
        % constructor
            self.L  = L;
            self.N  = N;
            self.nx = N;
            self.dx = 1 / self.N;
            self.V  = speye(self.N);
            self.x_init = zeros(self.N, 1);
            self.x_init(1) = 1;
        end

        function initialize(self)
            self.computeMatrices();
            self.initialized = true;
        end

        function computeMatrices(self)
        % discretization operators for first, second and fourth derivative
            e = ones(self.N,1);

            self.D1 = spdiags([e, 0*e, -1*e], -1:1, self.N, self.N);
            self.D1(1,self.N) = [1];
            self.D1(self.N,1) = [-1];
            self.D1 = self.D1 / (2*self.dx);

            self.D2 = spdiags([e, -2*e, e], -1:1, self.N, self.N);
            self.D2(1,self.N) = [1];
            self.D2(self.N,1) = [1];
            self.D2 = self.D2 / (self.dx^2);

            self.D4 = spdiags([e, -4*e, 6*e, -4*e, e], -2:2, self.N, self.N);
            self.D4(1,self.N-1:self.N) = [1, -4];
            self.D4(2,self.N)          = [1];
            self.D4(self.N-1,1)        = [1];
            self.D4(self.N,1:2)        = [-4, 1];
            self.D4 = self.D4 / (self.dx^4);

            self.Jlin = ((1+self.epsilon)/self.L^2)*self.D2 + (1/self.L^4)*self.D4;
        end

        function [out] = f(self, y)
        % RHS
            assert(numel(y) == self.N);
            assert(self.initialized, 'KSmodel not initialized');

            out = - (1/self.L)*y.*(self.D1*y) ...
                  - ((1+self.epsilon)/self.L^2)*self.D2*y ...
                  - (1/self.L^4)*self.D4*y;
        end

        function [out] = g(self, yp, ym, dt, frc)
        % time dependent rhs (backward Euler)

            assert(numel(frc) == self.N);
            out = yp - ym - dt * self.V'*(self.f(self.V*yp) + frc);
        end

        function [out] = J(self, y)
        % Jacobian
            assert(numel(y) == self.N);
            assert(self.initialized, 'KSmodel not initialized');

            dydx = self.D1*y;
            D1y  = sparse(1:self.N, 1:self.N, dydx, self.N, self.N);
            yD1  = (sparse(1:self.N, [self.N, 1:self.N-1], y) + ...
                    sparse(1:self.N, [2:self.N, 1], -y)) / (2*self.dx);

            out  =  -(1/self.L)*(yD1 + D1y) ...
                    -self.Jlin;
        end

        function [out] = J_old(self, y)
        % Jacobian (deprecated implementation)
            out = -(1/self.L)*(y.*self.D1+sparse(diag(self.D1*y))) ...
                  -(1/self.L^2)*self.D2 -(1/self.L^4)*self.D4;
        end

        function [out] = H(self, y, dt)
        % time dependent Jacobian (backward Euler)
            out = speye(min(size(self.V,2), self.N)) - dt*self.V'*self.J(self.V*y)*self.V;
        end

        function [y, k] = step(self, y, dt, frc)
        % perform single step time integration
            ym = y;

            if nargin < 4
                frc = zeros(self.N,1);
            end

            % Newton
            for k = 1:self.Nkmx
                H = self.H(y, dt);
                g = self.g(y, ym, dt, frc);
                dy = H \ -g;
                y  = y + dy;
                
                if (norm(dy) < self.Ntol)
                    break;
                end
            end
            if k == self.Nkmx
                ME = MException('KS:convergenceError', ...
                                'no convergence in Newton iteration');
                throw(ME);
            end
        end

        function [par] = control_param(self)
        % return control parameter
            par = self.epsilon;
        end
    end
end

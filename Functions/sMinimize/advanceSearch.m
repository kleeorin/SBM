function [x_out,t_out,h_out,g_out,gtd_out,s_out,y_out,ys_out,hess_diag_out,ind,skipping] = advanceSearch(x,t,h,g,fun,gtd,s,y,ys,hess_diag,i,options,ind,skipping)
    t_out=t;
    flag=1;
    while flag>0
        x_out = x + t_out*h;
            % calculate new point characteristics
        [g_out]=fun(x_out);
        [h_out,s_out,y_out,ys_out,hess_diag_out,ind,skipping]=UpdateHessian(g_out,g_out-g,x_out-x,s,y,ys,hess_diag,options,ind,skipping);
        gtd_out=-g_out'*h_out; 
        if gtd_out<options.gtdTol*gtd
            flag=0;
        end
    end

end


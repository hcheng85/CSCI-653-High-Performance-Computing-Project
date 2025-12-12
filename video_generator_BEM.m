 
    %%%% this is the lastest version of video_generator_BEM.m
    %%%% modified at Sep 4th
    % profile on 
    close all
    clc
    % clear 
    fig = figure('position',[1200 1200 720 720],'color','white');
    ratio = fig.Position(3)/fig.Position(4);
    ax = axes('position',[0 0 1 1]); hold all;
    global_ratio = 0.5;
    view_range = 25*global_ratio;
    xprop = 0.5;
    yprop = 0.5;
    
    addpath('../../Func');
    axis([-xprop*view_range (1-xprop)*view_range ...
        -yprop*view_range/ratio yprop*view_range/ratio]);
    axis equal
    
    saving = 0; %save or not
    warning off
    
    R = 5;
    n_sims = 3;
    num = 1;
    color_see = [0.97,0.78,0.78];
    color_not_see = [0.35, 0.35, 0.38];
    % file = ['If10_Ip40_In05_R',num2str(radius_ratio),'_num',num2str(num),'_gap',...
    %     num2str(gap_ratio),'_n',num2str(n_sims),'_BEM'];
    % load(['BEM_data/',file,'.mat']);
    % load([pwd,'/BEM_data/',file]);
    load(['geo_data/BEM_geometry_complex_enviornment_R',num2str(R) ...
        ,'.mat']);

    % idx_start = 1;  
    % for i = 1:length(Nper)
    %     idx_end = idx_start + Nper(i) - 1;
    %     plot([Sx(idx_start:idx_end);Sx(idx_start)], [Sy(idx_start:idx_end);Sy(idx_start)], 'k-', 'LineWidth', 1.2);
    %     idx_start = idx_end + 1;
    % end
    c = [0, cumsum(Nper)];
    Xgeom = []; Ygeom = [];
    Xgeom_cap = 0;  % optional reserve if you want to preallocate
    
    for k = 1:numel(Nper)
        seg = (c(k)+1):c(k+1);
        % close each loop and add NaN break
        Xgeom = [Xgeom; Sx(seg); Sx(seg(1)); NaN];
        Ygeom = [Ygeom; Sy(seg); Sy(seg(1)); NaN];
    end
    % 
    % fill(Xgeom, Ygeom, [0.1 0.8 0.8], 'EdgeColor', 'k', 'LineWidth', 1.2);
    for k = 1:numel(Nper)              
        if k==3 
            continue
        end
        seg = (c(k)+1):c(k+1);
        xv = [Sx(seg); Sx(seg(1))];       % close polygon
        yv = [Sy(seg); Sy(seg(1))];
        hfill = fill(xv, yv, [0.9 0.9 0.9], ...
                     'EdgeColor','k','LineWidth',1.2,'FaceAlpha',1);
        uistack(hfill,'bottom');          % keep below fish/trajectories
    end
    plot(Xgeom,Ygeom,'k-',LineWidth=1.2);
    if size(state,1) ~= double(para.num)
        state = pagetranspose(state);
        rdot = pagetranspose(rdot);
    end

    state(:,3,:) = wrapToPi(state(:,3,:));
    root = pwd;
    % folder = [root,'/images/',file];
    if  saving == 1 && exist(folder,'file') == 0
        mkdir(folder);
    end
    % plot(-1.2,4.7,'o')
    
    % ==================  TRAJECTORY OVERLAY  ==================
    fish_id     = 1;          % which fish to plot
    trail_secs  = 5;         % how long the trail shows (simulation seconds)
    trail_len   = max(2, round(trail_secs/para.dt));   % frames in the window
    
    % pre-create line handles for performance
    hBlue  = plot(NaN, NaN, '-', 'Color', color_see, 'LineWidth',2); % sees object
    hBlack = plot(NaN, NaN, '-', 'Color', color_not_see, 'LineWidth', 2); % cannot see
    % hNow   = plot(NaN, NaN, 'o', 'MarkerFaceColor', [1 0 0], ...
    %                         'MarkerEdgeColor', 'none', 'MarkerSize', 5); % current pos
    % ===========================================================

    % scatter(-1.2,4.7,60,'red','filled')
    scatter(target_pos(1),target_pos(2),60,'red','filled');
    EndTime = para.total_time; % dimensionless seconds%
    EndTimeSimulation = EndTime/double(para.dt);
    % StartTime = double(para.dt); % dimensionless seconds%
    % StartTime =para.dt;
    StartTime=para.dt;
    StartTimeSimulation = StartTime/double(para.dt);
    
    [pth,xfish,yfish] = FishVisualization(para,global_ratio);
    kkk = 1;
    cc = 1;
    % load('map1.mat');

    for i = StartTimeSimulation:StartTimeSimulation+EndTimeSimulation
        
        if saving == 1
            file_name = sprintf('%06d',kkk);
            % video_file_loc = [pwd,'/',folder,'/'];
            full_file_name_video = fullfile(folder,file_name);
        end
    
        ax.Position = ([0 0 1 1]);
        Cur_state = state(:,:,i);
        Cur_loc = Cur_state(:,1:2);
        Cur_rdot = rdot(:,:,i);
        e = [cos(Cur_state(:,3)) sin(Cur_state(:,3))];
        % if cc == 1
        %     Line = {};
        %     for b = 1:length(id_informed)
        %         Line{b} = plot(Cur_loc(id_informed(b),1), ...
        %             Cur_loc(id_informed(b),1));
        %         set(Line{b},Color=[0.97,0.78,0.78],LineWidth=2);
        %         hold on
        %     end
        % end
        % v = sqrt(sum(Cur_rdot.^2,2));
        % P(i) = norm(mean(e));    
        if mod(i,100) == 0
            % i*double(para.dt)
          
            %%%%%%%% plot voronoi neighbours %%%%%%%%%%%
            % if cc ~= 1
            %     for a = 1:length(Line_voro)
            %         delete(Line_voro{a});
            %     end
            % end
            % Line_voro = {};
            % % diploc = [dat_loc(1:para.num,i),dat_loc(para.num+1:end,i)];
            % [voro] = voronoi_neighbor(Cur_loc);
            % voro_loc = Cur_loc.*voro(id_informed,:)';
            % index_voro = find(isnan(voro(:,id_informed)) == 0);
            % for ii = 1:length(index_voro)
            %     Line_voro{ii} = plot([Cur_loc(id_informed,1),Cur_loc(index_voro(ii),1)], ...
            %         [Cur_loc(id_informed,2),Cur_loc(index_voro(ii),2)],'-','Visible','on', ...
            %         'LineWidth',1.5,'Color',[0.8 0.8 0.8]);
            % end

            xdisp = xfish.*e(:,1)' - yfish.*e(:,2)' + Cur_loc(:,1)';
            ydisp = xfish.*e(:,2)' + yfish.*e(:,1)' + Cur_loc(:,2)'; 

            set(pth,'XData',xdisp,'YData',ydisp);
            
            % --------- update visibility-colored trajectory ---------
            t0 = max(StartTimeSimulation, i - trail_len + 1);
            tW = t0:i;
    
            xs  = squeeze(state(fish_id,1,tW));
            ys  = squeeze(state(fish_id,2,tW));
            vis = squeeze(targetsearch(1,1,tW)) ~= 0;  % 1=sees, 0=not
    
            % Use NaNs to break the line at state switches
            x_blue = xs;  y_blue = ys;
            x_blue(~vis) = NaN; y_blue(~vis) = NaN;
    
            x_blk  = xs;  y_blk  = ys;
            x_blk(vis)  = NaN;   y_blk(vis)  = NaN;
    
            set(hBlue,  'XData', x_blue, 'YData', y_blue);
            set(hBlack, 'XData', x_blk,  'YData', y_blk);
    
            % current position marker
            % set(hNow, 'XData', xs(end), 'YData', ys(end));
    
            % --------------------------------------------------------

            axis off
            axis([-xprop*view_range (1-xprop)*view_range ...
                -yprop*view_range/ratio yprop*view_range/ratio]);
            drawnow
            kkk = kkk+1;
            
            if saving == 1
                print(gcf, full_file_name_video, '-dpng','-r400');
                % hgexport(gcf, full_file_name_video, hgexport('factorystyle'), 'Format', 'png');
            end       
            % profile viewer
            % print -depsc -painters fish_school.eps
        end
        progressbar(double(i-StartTimeSimulation),double(EndTimeSimulation-StartTimeSimulation+1));
    end
    
    
    function [pch,xfish,yfish] = FishVisualization(para,ratio)
    x = [1, 0.997326307812780, 0.989329653771421, 0.976083494661845, 0.957710815542293, ...
        0.934384725438304, 0.906328871134569, 0.873817371767845, 0.837174029901309, ...
        0.796770676365455, 0.753024625416443, 0.706395318588172, 0.657380290984390, ...
        0.606510588547310, 0.554345704569574, 0.501468012628056, 0.448476587514454, ...
        0.395900399313799, 0.343463287365574, 0.292760197966619, 0.244446145187294, ...
        0.199146319756159, 0.157440776608933, 0.119850695824284, 0.0868274376618654, ...
        0.0587452381740560, 0.0358978488122869, 0.0184988244222952, 0.00668463255766389, ...
        0.000519400107631375, 0]';

    y = [-2.06340757122885e-17, 0.000856822412222475, 0.00339805446150660, ...
        0.00753769904005638, 0.0131379054557917, 0.0200164635008182, ...
        0.0279559765592272, 0.0367136690047067, 0.0460308228498145, ...
        0.0556410479614177, 0.0652769299574405, 0.0746750056880809, ...
        0.0835794089842565, 0.0917448297777561, 0.0989395733342973, ...
        0.104949457151418, 0.109583042146393, 0.112676014594602, ...
        0.113675527905528, 0.112235265071430, 0.108426397502748, ...
        0.102404597845165, 0.0944042226848559, 0.0847267799978143, ...
        0.0737238876826811, 0.0617756084403257, 0.0492656514979399, ...
        0.0365553800287385, 0.0239588007690234, 0.0117207166825021, 0]';

    scale = 0.6;
    xfish = scale*x-0.4;
    yfish = scale*flipud(y);
    xfish = [xfish;flipud(xfish)]*ratio;
    yfish = [yfish;-flipud(yfish)]*ratio;

    % pch = patch(xfish+zeros(1,para.num-n),yfish+zeros(1,para.num-n),zeros(para.num-n,1),...
    %     'edgecolor','black','visible','on','LineWidth',1e-5); hold on
    pch = patch(xfish+zeros(1,para.num),yfish+zeros(1,para.num),zeros(para.num,1),...
        'facecolor','black','visible','on'); hold on
    % pch = patch(xfish+zeros(1,para.num-n),yfish+zeros(1,para.num-n),zeros(para.num-n,1),...
    %     'facecolor','black','edgecolor','none','visible','on'); hold on
    % if para.G == 0
    %     pinf = patch(xfish_inf+zeros(1,n),yfish_inf+zeros(1,n),zeros(n,1),...
    %         'facecolor','black','edgecolor','none','visible','on');
    % else
    %     pinf = patch(xfish_inf+zeros(1,n),yfish_inf+zeros(1,n),zeros(n,1),...
    %         'facecolor','red','edgecolor','none','visible','on');
    % end
    
    end



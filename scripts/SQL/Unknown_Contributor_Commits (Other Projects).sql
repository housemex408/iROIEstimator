BEGIN
  DECLARE projects ARRAY<STRING>;
  DECLARE task_types ARRAY<STRING>;
  DECLARE p_count INT64;
  DECLARE t_count INT64;
  DECLARE p_index INT64;
  DECLARE t_index INT64;
  DECLARE p_next STRING;
  DECLARE t_next STRING;
  
  SET task_types = [
  "BUG", "DOCS", "REFACTOR", "TESTING", "FEATURE", "UPGRADE", "RELEASE", "SUPPORT", "OTHER"
  ];
  SET projects = 
  [
    "d3/d3"
    ,"socketio/socket.io"
    ,"webpack/webpack"
    ,"jekyll/jekyll"
    ,"kennethreitz/requests"
    ,"lodash/lodash"
    ,"typicode/json-server"
    ,"Unitech/pm2"
    ,"videojs/video.js"
    ,"plataformatec/devise"
    ,"facebook/jest"
    ,"webtorrent/webtorrent"
    ,"npm/npm"
    ,"Automattic/mongoose"
    ,"visionmedia/superagent"
    ,"keystonejs/keystone"
    ,"yabwe/medium-editor"
    ,"jgm/pandoc"
    ,"swagger-api/swagger-ui"
    ,"SeleniumHQ/selenium"
    ,"BrowserSync/browser-sync"
    ,"jhipster/generator-jhipster"
    ,"karma-runner/karma"
    ,"jsdom/jsdom"
    ,"stylus/stylus"
    ,"NetEase/pomelo"
    ,"systemjs/systemjs"
    ,"jordansissel/fpm"
    ,"nightwatchjs/nightwatch"
    ,"danialfarid/ng-file-upload"
    ,"NaturalNode/natural"
    ,"datproject/dat"
    ,"middleman/middleman"
    ,"fluent/fluentd"
    ,"zloirock/core-js"
    ,"vega/vega"
    ,"josdejong/mathjs"
    ,"AutoMapper/AutoMapper"
  ];
  
  SET p_count = ARRAY_LENGTH(projects);
  SET t_count = ARRAY_LENGTH(task_types);
  SET p_index = 0;
  SET t_index = 0;
  
  WHILE p_index < p_count DO
    SET p_next = projects[OFFSET(p_index)];
      WHILE t_index < t_count DO
        SET t_next = task_types[OFFSET(t_index)];
        
        CREATE TABLE IF NOT EXISTS `praxis.uk_commits`
        (Project String, Version String, Date Date, Task String, T_Module INT64, T_Line INT64, N_O INT64, N_T INT64, Module INT64, Line INT64, Contributors INT64)
        OPTIONS(
          expiration_timestamp=TIMESTAMP "2021-01-01 00:00:00 UTC",
          description="Export external contributor commit tasks.  Expires in 2021"
        );
        INSERT INTO `praxis.uk_commits`
        SELECT rc.key as Project, rc.Version, vm.Release_Date as Date, tt.task as Task, vm.T_Module, vm.T_Line, ott.N_O,
        count(tt.task) as N_T, SUM(rc.E_Module) as Module, SUM(rc.E_Line) as Line, -1 as Contributors 
        FROM `gwu-praxis.praxis.ghtorrent_projects` as p
        INNER JOIN `gwu-praxis.praxis.repo_repository_commits` as rc
        ON SUBSTR(p.url, 30) = rc.Key
        INNER JOIN `praxis.repo_version_metrics` as vm
        ON vm.Key = rc.Key AND vm.Version = rc.Version
        INNER JOIN `praxis.task_types` as tt
        ON tt.Key = rc.Key AND tt.Version = rc.Version AND tt.SHA = rc.sha
        LEFT JOIN `gwu-praxis.praxis.ghtorrent_commits` as c
        ON c.sha = rc.SHA AND c.project_id = p.id
        LEFT JOIN 
        (
          SELECT tt2.key, tt2.version, count(tt2.task) as N_O 
          FROM `praxis.task_types` as tt2 
          LEFT JOIN `gwu-praxis.praxis.ghtorrent_commits` as c2
          ON c2.sha = tt2.SHA
          WHERE c2.commiter_id IS NULL
          AND tt2.task != t_next
          AND tt2.key = p_next
          GROUP BY tt2.key, tt2.version
        ) as ott
        ON ott.Key = rc.Key AND ott.version = rc.version
        WHERE c.commiter_id IS NULL
        AND rc.Key = p_next
        AND tt.task = t_next
        GROUP BY rc.key, rc.Version, vm.Release_Date, tt.task, vm.T_Module, vm.T_Line, ott.N_O
        ORDER BY vm.Release_Date ASC,  vm.T_Line ASC;

        SET t_index = t_index + 1;
      END WHILE;
    SET t_index = 0;
    SET p_index = p_index + 1;
  END WHILE;
END;
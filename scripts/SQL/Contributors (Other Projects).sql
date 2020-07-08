BEGIN
  DECLARE projects ARRAY<STRING>;
  DECLARE count INT64;
  DECLARE index INT64;
  DECLARE next STRING;
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

  SET count = ARRAY_LENGTH(projects);
  SET index = 0;
  
  WHILE index < count DO
    SET next = projects[OFFSET(index)];
    
    CREATE TEMP TABLE ALL_CONTRIBUTORS(project_id INT64, committer_id INT64, committer_login STRING);
    INSERT INTO ALL_CONTRIBUTORS(project_id, committer_id, committer_login)
    SELECT DISTINCT p.id as project_id, u.id as committer_id, u.login as committer_login
        FROM `praxis.ghtorrent_commits` as c, 
        `praxis.ghtorrent_users` as u, 
        `praxis.ghtorrent_projects` as p
        WHERE u.id = c.commiter_id
        AND u.fake = 0
        AND c.project_id = p.id
        AND SUBSTR(p.url, 30) = next;
     
     CREATE TEMP TABLE CORE_CONTRIBUTORS(project_id INT64, committer_id INT64, committer_login STRING);
     INSERT INTO CORE_CONTRIBUTORS(project_id, committer_id, committer_login)
     SELECT DISTINCT p.id as project_id, u.id as committer_id, u.login as committer_login
         FROM `praxis.ghtorrent_pull_requests` as pr, 
         `praxis.ghtorrent_projects` as p,
         `praxis.ghtorrent_users` as u, 
         `praxis.ghtorrent_pull_request_history` prh
         WHERE u.id = prh.actor_id
         AND prh.action = 'merged'
         AND prh.pull_request_id = pr.id
         AND pr.base_repo_id = p.id
         AND SUBSTR(p.url, 30) = next;
     
     CREATE TEMP TABLE EXTERNAL_CONTRIBUTORS(project_id INT64, committer_id INT64, committer_login STRING);
     INSERT INTO EXTERNAL_CONTRIBUTORS(project_id, committer_id, committer_login)
     SELECT * from ALL_CONTRIBUTORS AS ac
        WHERE ac.committer_id NOT IN (
          SELECT cc.committer_id FROM CORE_CONTRIBUTORS as cc
        );
     
     INSERT INTO `praxis.core_contributors`
     SELECT * FROM CORE_CONTRIBUTORS;
     
     INSERT INTO `praxis.external_contributors`
     SELECT * FROM EXTERNAL_CONTRIBUTORS;

     DROP TABLE ALL_CONTRIBUTORS;
     DROP TABLE CORE_CONTRIBUTORS;
     DROP TABLE EXTERNAL_CONTRIBUTORS;
     
    SET index = index + 1;
  END WHILE;
END;
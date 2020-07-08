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
    
    CREATE TEMP TABLE BUGS (Key STRING, Version STRING, SHA STRING, Task STRING);

    INSERT INTO BUGS(Key, Version, SHA, Task)
    SELECT Key, Version, SHA, "BUG" FROM `praxis.repo_commit_messages` as msg
    WHERE REGEXP_CONTAINS(LOWER(message), r'(bug|defect|fix)')
    AND msg.Key = next;
    
    INSERT INTO `praxis.task_types`
    SELECT * FROM BUGS;

    CREATE TEMP TABLE DOCS (Key STRING, Version STRING, SHA STRING, Task STRING);

    INSERT INTO DOCS(Key, Version, SHA, Task)
    SELECT Key, Version, SHA, "DOCS" as Task FROM `praxis.repo_commit_messages` as msg
    WHERE REGEXP_CONTAINS(LOWER(message), r'(docs|documentation)')
    AND SHA NOT IN (SELECT SHA FROM BUGS)
    AND msg.Key = next;
    
    INSERT INTO `praxis.task_types`
    SELECT * FROM DOCS;

    CREATE TEMP TABLE REFACTOR (Key STRING, Version STRING, SHA STRING, Task STRING);

    INSERT INTO REFACTOR(Key, Version, SHA, Task)
    SELECT Key, Version, SHA, "REFACTOR" as Task FROM `praxis.repo_commit_messages` as msg
    WHERE REGEXP_CONTAINS(LOWER(message), r'(refactor|refactoring)')
    AND SHA NOT IN (SELECT SHA FROM BUGS)
    AND SHA NOT IN (SELECT SHA FROM DOCS)
    AND msg.Key = next;
    
    INSERT INTO `praxis.task_types`
    SELECT * FROM REFACTOR;

    CREATE TEMP TABLE TESTING (Key STRING, Version STRING, SHA STRING, Task STRING);

    INSERT INTO TESTING(Key, Version, SHA, Task)
    SELECT Key, Version, SHA, "TESTING" as Task FROM `praxis.repo_commit_messages` as msg
    WHERE REGEXP_CONTAINS(LOWER(message), r'(testing|test|qa)')
    AND SHA NOT IN (SELECT SHA FROM BUGS)
    AND SHA NOT IN (SELECT SHA FROM DOCS)
    AND SHA NOT IN (SELECT SHA FROM REFACTOR)
    AND msg.Key = next;
    
    INSERT INTO `praxis.task_types`
    SELECT * FROM TESTING;

    CREATE TEMP TABLE FEATURE (Key STRING, Version STRING, SHA STRING, Task STRING);

    INSERT INTO FEATURE(Key, Version, SHA, Task)
    SELECT Key, Version, SHA, "FEATURE" as Task FROM `praxis.repo_commit_messages` as msg
    WHERE REGEXP_CONTAINS(LOWER(message), r'(feat|feature|enhancement)')
    AND SHA NOT IN (SELECT SHA FROM BUGS)
    AND SHA NOT IN (SELECT SHA FROM DOCS)
    AND SHA NOT IN (SELECT SHA FROM REFACTOR)
    AND SHA NOT IN (SELECT SHA FROM TESTING)
    AND msg.Key = next;
    
    INSERT INTO `praxis.task_types`
    SELECT * FROM FEATURE;

    CREATE TEMP TABLE UPGRADE (Key STRING, Version STRING, SHA STRING, Task STRING);

    INSERT INTO UPGRADE(Key, Version, SHA, Task)
    SELECT Key, Version, SHA, "UPGRADE" as Task FROM `praxis.repo_commit_messages` as msg
    WHERE REGEXP_CONTAINS(LOWER(message), r'(install|upgrade)')
    AND SHA NOT IN (SELECT SHA FROM BUGS)
    AND SHA NOT IN (SELECT SHA FROM DOCS)
    AND SHA NOT IN (SELECT SHA FROM REFACTOR)
    AND SHA NOT IN (SELECT SHA FROM TESTING)
    AND SHA NOT IN (SELECT SHA FROM FEATURE)
    AND msg.Key = next;
    
    INSERT INTO `praxis.task_types`
    SELECT * FROM UPGRADE;

    CREATE TEMP TABLE RELEASE (Key STRING, Version STRING, SHA STRING, Task STRING);

    INSERT INTO RELEASE(Key, Version, SHA, Task)
    SELECT Key, Version, SHA, "RELEASE" as Task FROM `praxis.repo_commit_messages` as msg
    WHERE REGEXP_CONTAINS(LOWER(message), r'(release)')
    AND SHA NOT IN (SELECT SHA FROM BUGS)
    AND SHA NOT IN (SELECT SHA FROM DOCS)
    AND SHA NOT IN (SELECT SHA FROM REFACTOR)
    AND SHA NOT IN (SELECT SHA FROM TESTING)
    AND SHA NOT IN (SELECT SHA FROM FEATURE)
    AND SHA NOT IN (SELECT SHA FROM UPGRADE)
    AND msg.Key = next;
    
    INSERT INTO `praxis.task_types`
    SELECT * FROM RELEASE;

    CREATE TEMP TABLE SUPPORT (Key STRING, Version STRING, SHA STRING, Task STRING);

    INSERT INTO SUPPORT(Key, Version, SHA, Task)
    SELECT Key, Version, SHA, "SUPPORT" as Task FROM `praxis.repo_commit_messages` as msg
    WHERE REGEXP_CONTAINS(LOWER(message), r'(support|question|faq)')
    AND SHA NOT IN (SELECT SHA FROM BUGS)
    AND SHA NOT IN (SELECT SHA FROM DOCS)
    AND SHA NOT IN (SELECT SHA FROM REFACTOR)
    AND SHA NOT IN (SELECT SHA FROM TESTING)
    AND SHA NOT IN (SELECT SHA FROM FEATURE)
    AND SHA NOT IN (SELECT SHA FROM UPGRADE)
    AND SHA NOT IN (SELECT SHA FROM RELEASE)
    AND msg.Key = next;
    
    INSERT INTO `praxis.task_types`
    SELECT * FROM SUPPORT;

    CREATE TEMP TABLE OTHER (Key STRING, Version STRING, SHA STRING, Task STRING);

    INSERT INTO OTHER(Key, Version, SHA, Task)
    SELECT Key, Version, SHA, "OTHER" as Task FROM `praxis.repo_commit_messages` as msg
    WHERE SHA NOT IN (SELECT SHA FROM BUGS)
    AND SHA NOT IN (SELECT SHA FROM DOCS)
    AND SHA NOT IN (SELECT SHA FROM REFACTOR)
    AND SHA NOT IN (SELECT SHA FROM TESTING)
    AND SHA NOT IN (SELECT SHA FROM FEATURE)
    AND SHA NOT IN (SELECT SHA FROM UPGRADE)
    AND SHA NOT IN (SELECT SHA FROM RELEASE)
    AND SHA NOT IN (SELECT SHA FROM SUPPORT)
    AND msg.Key = next;

    INSERT INTO `praxis.task_types`
    SELECT * FROM OTHER;

    DROP TABLE BUGS;
    DROP TABLE DOCS;
    DROP TABLE REFACTOR;
    DROP TABLE TESTING;
    DROP TABLE FEATURE;
    DROP TABLE UPGRADE;
    DROP TABLE RELEASE;
    DROP TABLE SUPPORT;
    DROP TABLE OTHER;

    SET index = index + 1;
  END WHILE;
END;
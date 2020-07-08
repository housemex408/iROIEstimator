DECLARE project_name STRING;
DECLARE task_type STRING;

SET project_name = "angular/angular";
SET task_type = "BUG";

SELECT vm.Key as Project, vm.Version, vm.Release_Date as Date, uc.Task, vm.T_Module, vm.T_Line, 
cc.N_T as NT_CC, cc.N_O as NO_CC, cc.Module as Module_CC, cc.Line as Line_CC, cc.Contributors as T_CC,
ec.N_T as NT_EC, ec.N_O as NO_EC, ec.Module as Module_EC, ec.Line as Line_EC, ec.Contributors as T_EC,
uc.N_T as NT_UC, uc.N_O as NO_UC, uc.Module as Module_UC, uc.Line as Line_UC, uc.Contributors as T_UC,
ROUND(cc.Line / (cc.Line + ec.Line), 2) as Line_CC_P, 
ROUND(ec.Line / (cc.Line + ec.Line), 2) as Line_EC_P,
ROUND(cc.Module / (cc.Module + ec.Module), 2) as Module_CC_P, 
ROUND(ec.Module / (cc.Module + ec.Module), 2) as Module_EC_P
FROM `praxis.repo_version_metrics` as vm
LEFT JOIN 
( 
SELECT * 
FROM `praxis.ccc_commit_tasks`
WHERE Project = project_name
AND Task = task_type 
) AS cc
ON cc.Project = vm.Key AND cc.Version = vm.Version
LEFT JOIN
(
SELECT * 
FROM `praxis.ecc_commit_tasks`
WHERE Project = project_name
AND Task = task_type 
) AS ec
ON ec.Project = vm.Key AND ec.Version = vm.Version
LEFT JOIN
(
SELECT * 
FROM `praxis.uk_commit_tasks`
WHERE Project = project_name
AND Task = task_type 
) AS uc
ON uc.Project = vm.Key AND uc.Version = vm.Version
WHERE vm.Key = project_name

ORDER BY vm.Key, vm.Release_Date, vm.T_Line, IFNULL(cc.Task, 'Z') ASC, vm.Version DESC
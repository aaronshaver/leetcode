-- --------------------------------------------------------------------------
-- https://leetcode.com/problems/combine-two-tables/submissions/

select firstName, lastName, city, state
from Person p
left join Address a
on p.personId = a.personId

-- --------------------------------------------------------------------------
-- https://leetcode.com/problems/swap-salary/solution/

UPDATE salary
SET
    sex =
        CASE sex
	    WHEN 'm' THEN 'f'
	    WHEN 'f' THEN 'm'
	END;

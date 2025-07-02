-- 21-div.sql
-- Creates a function SafeDiv that divides a by b or returns 0 if b == 0
DELIMITER //
CREATE FUNCTION SafeDiv(a INT, b INT) RETURNS FLOAT
DETERMINISTIC
BEGIN
    IF b = 0 THEN
        RETURN 0;
    END IF;
    RETURN a / b;
END;//
DELIMITER ; 
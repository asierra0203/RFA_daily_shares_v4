SELECT
    t1.ship_code AS casino_code,
    t1.curr_number_sail_nights AS sailing_length,
    t1.meta_product_code,
    TO_CHAR(t1.sailing_date, 'YYYY-MM-DD') AS departure_date,
    LISTAGG(t2_filtered.port_code, ',') 
        WITHIN GROUP (ORDER BY t1.ship_code, t1.sailing_date, t2_filtered.berth_date) AS port_codes
FROM Rcihop.icslmd_companion_edss t1
JOIN (
    SELECT
        ship_code,
        sail_date,
        berth_date,
        port_code,
        departure_datetime
    FROM (
        SELECT
            ship_code,
            sail_date,
            berth_date,
            port_code,
            departure_datetime,
            ROW_NUMBER() OVER (
                PARTITION BY ship_code, sail_date
                ORDER BY berth_date DESC
            ) AS rn_desc,
            ROW_NUMBER() OVER (
                PARTITION BY ship_code, sail_date, berth_date
                ORDER BY departure_datetime DESC
            ) AS rn
        FROM mkrpdply.SHIP_ITIN_MASTER_AS400@edssp
    )
    WHERE rn = 1 AND rn_desc > 1 -- Excludes the last port for each sailing
) t2_filtered
    ON t1.sailing_date = t2_filtered.sail_date
    AND t1.ship_code = t2_filtered.ship_code
WHERE
    t1.valid_voyage = 'Y'
    AND t1.brand = 'R'
--    AND t1.sailing_date BETWEEN TRUNC(SYSDATE) AND TO_DATE('30-sep-25', 'DD-MON-YY')
GROUP BY
    t1.ship_code,
    t1.meta_product_code,
    TO_CHAR(t1.sailing_date, 'YYYY-MM-DD'),
    t1.curr_number_sail_nights
ORDER BY
    TO_CHAR(t1.sailing_date, 'YYYY-MM-DD');






function circle_line_intersecting(ax, ay, bx, by, cx, cy, r)
    area2 = abs((bx-ax) * (cy-ay) - (cx-ax) * (by-ay))
    lab = sqrt((bx-ax)^2 + (by-ay)^2)
    h = area2/lab
    h < r
end
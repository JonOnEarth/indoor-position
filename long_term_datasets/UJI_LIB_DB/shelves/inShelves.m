% Copyright © 2017 Universitat Jaume I (UJI)
%
% Permission is hereby granted, free of charge, to any person obtaining a copy of
% this software and associated documentation files (the “Software”), to deal in
% the Software without restriction, including without limitation the rights to
% use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
% of the Software, and to permit persons to whom the Software is furnished to do
% so, subject to the following conditions:
%
% The above copyright notice and this permission notice shall be included in all
% copies or substantial portions of the Software.
%
% THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
% IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
% FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
% AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
% LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
% OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
% SOFTWARE.

function [indexes] = inShelves(shelfPolys, positions)
% inShelves  Return indexes of positions (x,y) that lies within or at the
% boundary of the polygons shelfPolys (as returned from loadShelves method)
%
%   See also loadShelves, plotShelves.
    indexes = false(size(positions,1),1);
    for i = (1:size(shelfPolys,1)/4)
        polyData = shelfPolys(shelfPolys(:,1)==i,[2,3]);
        in = inpolygon(positions(:,1), positions(:,2), polyData(:,1), polyData(:,2));
        indexes = indexes | in;
    end
end
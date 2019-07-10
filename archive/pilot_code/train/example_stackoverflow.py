from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np
# setup Lambert Conformal basemap.
m = Basemap(width=12000000,height=9000000,projection='lcc',
            resolution='c',lat_1=45.,lat_2=55,lat_0=50,lon_0=-107.)

m.drawmapboundary(fill_color='azure')
m.fillcontinents(color='sandybrown',lake_color='azure')

parallels = np.arange(0.,81,10.)
m.drawparallels(parallels,labels=[False,True,True,False])
meridians = np.arange(10.,351.,20.)
m.drawmeridians(meridians,labels=[True,False,False,True])
# plot blue dot on Boulder, colorado and label it as such.
lon, lat = -104.237, 40.125 # Location of Boulder
# convert to map projection coords.
# Note that lon,lat can be scalars, lists or numpy arrays.
xpt,ypt = m(lon,lat)
# convert back to lat/lon
lonpt, latpt = m(xpt,ypt,inverse=True)
point, = m.plot(xpt,ypt,'bo')  # plot a blue dot there
# put some text next to the dot, offset a little bit
# (the offset is in map projection coordinates)
annotation = plt.annotate('Boulder (%5.1fW,%3.1fN)' % (lon, lat), xy=(xpt,ypt),
             xytext=(20,35), textcoords="offset points", 
             bbox={"facecolor":"w", "alpha":0.5}, 
             arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))

def onclick(event):
    ix, iy = event.xdata, event.ydata
    xpti, ypti = m(ix, iy,inverse=True)
    string = '(%5.1fW,%3.1fN)' % (xpti, ypti)
    print(string)
    annotation.xy = (ix, iy)
    point.set_data([ix], [iy])
    annotation.set_text(string)
    plt.gcf().canvas.draw_idle()

cid = plt.gcf().canvas.mpl_connect("button_press_event", onclick)
plt.show()
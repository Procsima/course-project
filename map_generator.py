import matplotlib.pyplot as plt
import cartopy.crs as ccrs

fig, ax = plt.subplots(figsize=(18, 9), subplot_kw={'projection': ccrs.PlateCarree()})

ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())

ax.coastlines()
ax.gridlines(draw_labels=False)

plt.show()

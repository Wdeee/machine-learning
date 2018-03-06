def binning(latitude1):
	a=[]
	b=[]
	for latitude in latitude1:
		if latitude>32.3 and latitude<=32.6:
			a.append(1)
		else:
			a.append(0)
		if latitude>32.6 and latitude<=32.9:
			a.append(1)
		else:
			a.append(0)
		if latitude>32.9 and latitude<=33.2:
			a.append(1)
		else:
			a.append(0)
		if latitude>33.2 and latitude<=33.5:
			a.append(1)
		else:
			a.append(0)	
		if latitude>33.5 and latitude<=33.8:
			a.append(1)
		else:
			a.append(0)	
		if latitude>33.8 and latitude<=34.1:
			a.append(1)
		else:
			a.append(0)
		if latitude>34.1 and latitude<=34.4:
			a.append(1)
		else:
			a.append(0)
		if latitude>34.4 and latitude<=34.7:
			a.append(1)
		else:
			a.append(0)
		if latitude>34.7 and latitude<=35.0:
			a.append(1)
		else:
			a.append(0)
		if latitude>35.0 and latitude<=35.3:
			a.append(1)
		else:
			a.append(0)	
		if latitude>35.3 and latitude<=35.6:
			a.append(1)
		else:
			a.append(0)
		b.append(a)
		a=[]
	return str(b)

a=(33.4,34.1,32.7,33.8)
print(binning(a))
if isinstance(binning(a),str)==True:
	print('是str类型')													
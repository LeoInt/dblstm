import os
import subprocess
import time


myevents='l2_subp0_total_read_sector_queries,l1_local_load_hit,l1_shared_load_transactions,l1_global_load_hit,l1_local_store_hit,l1_shared_store_transactions'
mymetrics='shared_efficiency,inst_integer,inst_bit_convert,inst_control,inst_misc,inst_fp_32,inst_fp_64'
ROOT="/home/ubuntu/dblstm/rnn/final_test/"
NVPROF="/usr/local/cuda-6.5/bin/nvprof"
os.system('echo ciao')
ResultsList=[]
#projects = ["standard", "fusepw","streamfwbw","streamLayer","halffloat","lowmem","fuse_gemvpw"]
#projects = ["fusepw","streamfwbw","halffloat","lowmem","fuse_gemvpw"]
projects = ["cpu"]
hiddensize = [128, 320, 512]
#hiddensize = [320, 512]
seqlength = [16, 100, 150, 200, 250, 300]
#seqlength = [150]#, 200, 250, 300]
layers = [2,3,4,5]
#layers = [5]

for project in projects:  	
	for sq in seqlength:
		for hl in layers:
			for hs in hiddensize:	
				outputfilename=project+'/results/output'+str(sq)+'_'+str(hl)+'_'+str(hs)+'.csv'				
				proffilename=project+'/results/prof'+str(sq)+'_'+str(hl)+'_'+str(hs)+'.txt'
				metricsfilename=project+'/results/metrics'+str(sq)+'_'+str(hl)+'_'+str(hs)+'.txt'
				eventsfilename=project+'/results/events'+str(sq)+'_'+str(hl)+'_'+str(hs)+'.txt'
				exist=True				
				exist=os.path.isfile(outputfilename)				
				if project != 'cpu':
					exist=exist and os.path.isfile(proffilename)
					if sq==16 and hl==2:
						exist = exist and os.path.isfile(metricsfilename) and os.path.isfile(eventsfilename)
				if exist == True and project != 'cpu':
					continue
				time.sleep(5)			
				record=project+','+str(sq)+','+str(hl)+','+str(hs)+','
				#prjdircmd='cd '+project+'\n'
				#print prjdircmd				
				#os.system(prjdircmd)				
				command='./'+project+'/app '+str(sq)+' '+str(hl)+' '+str(hs)+' > outputfile'
				#args=[]
				#args.append(command)
				#args.append(str(sq))
				#args.append(str(hl))
				#args.append(str(hs))			
				print command
				#print output
				exetime=0
				ltime=[]
				for run in range(3):
					#print run															
					os.system(command)
					time.sleep(5)				
					with open("outputfile") as f:
						ltime.append(float(f.read()))
					os.system("rm outputfile")
					
				#exetime=float(output)
				#ltime.append(exetime)	
				exetime=min(ltime)
				record=record+str(exetime)+"\n"
				ResultsList.append(record)					
				f1=open(outputfilename, 'w+') 				
				f1.write(record)
				f1.close()				
				print exetime
				if project != "cpu":
					command='nvprof --log-file '+project+'/results/prof'+str(sq)+'_'+str(hl)+'_'+str(hs)+'.txt '+'./'+project+'/app '+str(sq)+' '+str(hl)+' '+str(hs)					
					print command	
					os.system(command)				
					time.sleep(5)					
					if sq==16 and hl==2:					
						command='nvprof --events '+myevents+' --log-file '+project+'/results/events'+str(sq)+'_'+str(hl)+'_'+str(hs)+'.txt '+'./'+project+'/app '+str(sq)+' '+str(hl)+' '+str(hs)					
						time.sleep(5)						
						print command	
						os.system(command)
						command='nvprof --metrics '+mymetrics+' --log-file '+project+'/results/metrics'+str(sq)+'_'+str(hl)+'_'+str(hs)+'.txt '+'./'+project+'/app '+str(sq)+' '+str(hl)+' '+str(hs)					
						time.sleep(5)						
						print command	
						os.system(command)						
					
													
					
	#prjdircmd='cd ..'+' \n'	
	#os.system(prjdircmd)

#print ResultsList


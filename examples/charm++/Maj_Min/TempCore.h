static int cpufreq_sysfs_read (int proc)
{
        FILE *fd;
        char path[100];
        int i=proc;
        sprintf(path,"/sys/devices/system/cpu/cpu%d/cpufreq/scaling_setspeed",i);

        fd = fopen (path, "r");

        if (!fd) {
                printf("FILE OPEN ERROR file=%s\n",path);
                return 0;
        }
        char val[10];
        fgets(val,10,fd);
        int ff=atoi(val);
        fclose (fd);

        return ff;
}

//FILE *f;
void writeTemps(FILE *f,char *temps)
{
//        FILE *f;
//        f=fopen("temps.out","a+");
        fprintf(f,"%s\n",temps);
//        fclose(f);
}

float getTemp(int cpu)
{
        char val[10];
        FILE *f;
                char path[100];
                sprintf(path,"/sys/devices/platform/coretemp.%d/temp1_input",cpu);
                f=fopen(path,"r");
                if (!f) {
                        printf("FILE OPEN ERROR file=\n");
                        exit(0);
                }

        if(f==NULL) {printf("ddddddddddddddddddddddddddd\n");exit(0);}
        fgets(val,10,f);
        fclose(f);
        return atof(val)/1000;
}



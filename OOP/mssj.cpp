#include <bits/stdc++.h>
using namespace std;
typedef pair<int,int> pii;
const int N = 3000;
int M, n, k, T, R;
int red[5] = {2, 3, 4, 1, 0}, blu[5] = {3, 0, 1, 2, 4};
int atk[5], hp[5];
struct W{
    int type, used, atk;
    bool operator<(const W &b) const { return type < b.type || (type == b.type && used < b.used); }
    W(int TYPE, int ATK){used = 0; type=TYPE; atk = 0;if(TYPE==0)atk=ATK/5;}
};
bool cmp(W a, W b) { return a.type < b.type || (a.type == b.type && a.used > b.used); }
struct node{
    int id, type, atk, hp, location, camp, ID;
    bool dead, arrive;
    int loyalty, step;
    double morale;
    vector<W> weapon;
    bool operator<(const node &b) const{return (dead < b.dead) || (dead == b.dead && location < b.location) || (dead == b.dead && location == b.location && camp < b.camp);}
} s[500010];
void Time(int h, int m){
    cout<<setw(3)<<setfill('0')<<h<<":";
    cout<<setw(2)<<setfill('0')<<m<<" ";
}
int Flag[2], Hp[2], Numb, Numr, peo, num[3], Arr[2], Lose[2], numcity[10010];
char Camp[2][10] = {"red", "blue"};
char name[5][10] = {"dragon", "ninja", "iceman", "lion", "wolf"};
int aw[2];vector<pair<int,int> >awl[2];
int win[10010][2],flag[10010];
vector<W> TMP;
int city[N];
bool USED(W a){
    if (a.type == 1 && a.used >=3)return true;
    if (a.type == 2 && a.used >= 1)return true;
    return false;
}
void use_arrow(node&a){
    sort(a.weapon.begin(),a.weapon.end());
    vector<W>TMP;TMP.clear();
    for(auto tmp:a.weapon){
        if(tmp.type==2){
            tmp.used++;
            if(tmp.used!=3)TMP.push_back(tmp);
        }
        else TMP.push_back(tmp);
    }
    a.weapon=TMP;
}
bool have_arrow(node a){for(auto tmp:a.weapon)if(tmp.type==2&&tmp.used<3)return true;return false;}
bool have_bomb(node a){for(auto tmp:a.weapon)if(tmp.type==1)return true;return false;}
int sgn(int cmp){return cmp==0?1:-1;}
int sword(node a){
    for(auto tmp:a.weapon)
        if(tmp.type==0)return tmp.atk;
    return 0;
}
bool attack(node a,node b,int tmp){
    int c=a.location;
    if(tmp){
        int ATK=a.atk+sword(a);
        b.hp-=ATK;
        if(b.hp>0&&b.type!=1){
            ATK=b.atk/2+sword(b);
            a.hp-=ATK;
        }
    }
    else{
        int ATK=b.atk+sword(b);
        a.hp-=ATK;
        if(a.hp>0&&a.type!=1){
            ATK=a.atk/2+sword(a);
            b.hp-=ATK;
        }
    }
    return a.hp<=0;
}
int nowH = 0, nowT = 0, it1 = 0, it2 = 0;
void SWORD(node &a){
    vector<W>remo;remo.clear();int FLAG=0;
    for(auto tmp:a.weapon)if(tmp.type==0){tmp.atk=tmp.atk*4/5;if(tmp.atk!=0)remo.push_back(tmp);}
    else remo.push_back(tmp);
    a.weapon=remo;
}
bool CMP(node a,node b){
    return a.dead<b.dead||(a.dead==b.dead&&a.camp<b.camp)||(a.dead==b.dead && a.camp==b.camp&&a.location<b.location);
}
void real_attack(node&a,node&b){
    int c=a.location;
    if(flag[c]==sgn(a.camp)||(!flag[c]&&(c&1))){
        int ATK=a.atk+sword(a);
        b.hp-=ATK;SWORD(a);
        Time(nowH,nowT);printf("%s %s %d attacked %s %s %d in city %d with %d elements and force %d\n",Camp[a.camp],name[a.type],a.id,Camp[b.camp],name[b.type],b.id,c,a.hp,a.atk);
        if(b.hp>0&&b.type!=1){
            ATK=b.atk/2+sword(b);
            a.hp-=ATK;SWORD(b);
            Time(nowH,nowT);printf("%s %s %d fought back against %s %s %d in city %d\n",Camp[b.camp],name[b.type],b.id,Camp[a.camp],name[a.type],a.id,c);
        }
    }
    else{
        int ATK=b.atk+sword(b);
        a.hp-=ATK;SWORD(b);
        Time(nowH,nowT);printf("%s %s %d attacked %s %s %d in city %d with %d elements and force %d\n",Camp[b.camp],name[b.type],b.id,Camp[a.camp],name[a.type],a.id,c,b.hp,b.atk);
        if(a.hp>0&&a.type!=1){
            ATK=a.atk/2+sword(a);
            b.hp-=ATK;SWORD(a);
            Time(nowH,nowT);printf("%s %s %d fought back against %s %s %d in city %d\n",Camp[a.camp],name[a.type],a.id,Camp[b.camp],name[b.type],b.id,c);
        }
    }
}
bool check(node&a,node&b,int c){
    if(a.hp<=0 && b.hp<=0){
        //win[c][0]=win[c][1]=0;
        return false;
    }
    if(a.hp<=0&&b.hp>0){
        win[c][b.camp]++;win[c][a.camp]=0;
        // printf("%d:%d %d\n",c,win[c][0],win[c][1]);
        aw[b.camp]+=city[c];
        if(b.type==4){
            num[0]=num[1]=num[2]=0;
            for(auto tmp:b.weapon)num[tmp.type]++;
            for(auto tmp:a.weapon)
                if(num[tmp.type]==0)b.weapon.push_back(tmp);
            sort(b.weapon.begin(),b.weapon.end());
            a.weapon.clear();
        }
        if(b.type==0){
            b.morale+=0.2;
            if(b.morale>=0.7&&(flag[c]==-1||(!flag[c]&&c%2==0))){
                Time(nowH,nowT);
                printf("%s %s %d yelled in city %d\n",Camp[b.camp],name[b.type],b.id,c);
            }
        }
        Time(nowH,nowT);printf("%s %s %d earned %d elements for his headquarter\n",Camp[b.camp],name[b.type],b.id,city[c]);
        awl[b.camp].push_back(make_pair(c,b.ID));
        city[c]=0;return false;
    }
    if(a.hp>0&&b.hp<=0){
        win[c][a.camp]++;win[c][b.camp]=0;
        // printf("%d:%d %d\n",c,win[c][0],win[c][1]);
        aw[a.camp]+=city[c];
        if(a.type==4){
            num[0]=num[1]=num[2]=0;
            for(auto tmp:a.weapon)num[tmp.type]++;
            for(auto tmp:b.weapon)
                if(num[tmp.type]==0)a.weapon.push_back(tmp);
            sort(a.weapon.begin(),a.weapon.end());
            b.weapon.clear();
        }
        if(a.type==0){
            a.morale+=0.2;
            if(a.morale>=0.7&&(flag[c]==1||(!flag[c]&&(c&1)))){
                Time(nowH,nowT);
                printf("%s %s %d yelled in city %d\n",Camp[a.camp],name[a.type],a.id,c);
            }
        }
        Time(nowH,nowT);printf("%s %s %d earned %d elements for his headquarter\n",Camp[a.camp],name[a.type],a.id,city[c]);
        awl[a.camp].push_back(make_pair(c,a.ID));
        city[c]=0;return false;
    }
    return true;
}
bool cmp1(pii a,pii b){return a.first>b.first;}
bool cmp2(pii a,pii b){return a.first<b.first;}
void work(){
    Flag[0] = Flag[1] = 1;
    Arr[0] = Arr[1] = 0;
    Lose[0] = Lose[1] = 0;
    nowH = 0, nowT = 0, it1 = 0, it2 = 0;
    scanf("%d%d%d%d%d", &M, &n, &R, &k, &T);
    Hp[0] = Hp[1] = M;
    for (int i = 0; i < 5; i++)scanf("%d", &hp[i]);
    for (int i = 0; i < 5; i++)scanf("%d", &atk[i]);
    while (T >= 0 && Lose[0]<2 && Lose[1]<2){
        if (nowT == 0){
            if (Hp[0] >= hp[red[it1]]){
                Time(nowH, nowT);
                printf("%s %s %d born\n", Camp[0], name[red[it1]], ++Numr);
                s[++peo] = (node){Numr, red[it1], atk[red[it1]], hp[red[it1]], 0, 0};
                s[peo].ID=peo; s[peo].weapon.clear();s[peo].dead=s[peo].arrive=false;s[peo].step=0;
                Hp[0] -= hp[red[it1]];
                if (red[it1] == 0){
                    s[peo].weapon.push_back(W(s[peo].id % 3,s[peo].atk));
                    if(s[peo].weapon.back().atk==0&&s[peo].weapon.back().type==0)s[peo].weapon.pop_back();
                    s[peo].morale = Hp[0] * 1. / hp[0];
                    printf("Its morale is %.2lf\n",s[peo].morale);
                }
                if (red[it1] == 1){
                    s[peo].weapon.push_back(W(s[peo].id % 3,s[peo].atk));
                    if(s[peo].weapon.back().atk==0&&s[peo].weapon.back().type==0)s[peo].weapon.pop_back();
                    s[peo].weapon.push_back(W((s[peo].id + 1) % 3,s[peo].atk));
                    if(s[peo].weapon.back().atk==0&&s[peo].weapon.back().type==0)s[peo].weapon.pop_back();
                }
                if (red[it1] == 2){
                    s[peo].weapon.push_back(W(s[peo].id % 3,s[peo].atk));
                    if(s[peo].weapon.back().atk==0&&s[peo].weapon.back().type==0)s[peo].weapon.pop_back();
                }
                if (red[it1] == 3){
                    printf("Its loyalty is %d\n", Hp[0]);
                    s[peo].loyalty = Hp[0];
                }
                it1 = (it1 + 1) % 5;
            }
            if (Hp[1] >= hp[blu[it2]]){
                Time(nowH, nowT);
                printf("%s %s %d born\n", Camp[1], name[blu[it2]], ++Numb);
                Hp[1] -= hp[blu[it2]];
                s[++peo] = (node){Numb, blu[it2], atk[blu[it2]], hp[blu[it2]], n + 1, 1};
                s[peo].ID=peo; s[peo].weapon.clear();s[peo].dead=s[peo].arrive=false;s[peo].step=0;
                if (blu[it2] == 0){
                    s[peo].weapon.push_back(W(s[peo].id % 3,s[peo].atk));
                    if(s[peo].weapon.back().atk==0&&s[peo].weapon.back().type==0)s[peo].weapon.pop_back();
                    s[peo].morale = Hp[1] * 1. / hp[0];
                    printf("Its morale is %.2lf\n",s[peo].morale);
                }
                if (blu[it2] == 1){
                    s[peo].weapon.push_back(W(s[peo].id % 3,s[peo].atk));
                    if(s[peo].weapon.back().atk==0&&s[peo].weapon.back().type==0)s[peo].weapon.pop_back();
                    s[peo].weapon.push_back(W((s[peo].id + 1) % 3,s[peo].atk));
                    if(s[peo].weapon.back().atk==0&&s[peo].weapon.back().type==0)s[peo].weapon.pop_back();
                }
                if (blu[it2] == 2){
                    s[peo].weapon.push_back(W(s[peo].id % 3,s[peo].atk));
                    if(s[peo].weapon.back().atk==0&&s[peo].weapon.back().type==0)s[peo].weapon.pop_back();
                }
                if (blu[it2] == 3){
                    printf("Its loyalty is %d\n", Hp[1]);
                    s[peo].loyalty = Hp[1];
                }
                it2 = (it2 + 1) % 5;
            }
        }
        if (nowT == 5){
            sort(s + 1, s + peo + 1);
            for (int i = 1; i <= peo; i++)
                if (!s[i].dead && s[i].type == 3 && s[i].loyalty <= 0 && !s[i].arrive){
                    Time(nowH, nowT);
                    printf("%s lion %d ran away\n", Camp[s[i].camp], s[i].id);
                    s[i].dead = 1;
                }
        }
        if (nowT == 10){
            for (int i = 1; i <= peo; i++)
                if (!s[i].dead && !s[i].arrive){
                    s[i].step++;
                    if (s[i].camp){
                        s[i].location--;
                        if (s[i].type == 2&&s[i].step%2==0){s[i].hp=max(1,s[i].hp-9);s[i].atk+=20;}
                    }
                    else{
                        s[i].location++;
                        if (s[i].type == 2&&s[i].step%2==0){s[i].hp=max(1,s[i].hp-9);s[i].atk+=20;}
                    }
                }
            sort(s + 1, s + peo + 1);
            for (int i = 1; i <= peo; i++)
                if (!s[i].arrive && !s[i].dead){
                    Time(nowH, nowT);
                    if (s[i].location != n + 1 && s[i].location != 0)printf("%s %s %d marched to city %d with %d elements and force %d\n", Camp[s[i].camp], name[s[i].type], s[i].id, s[i].location, s[i].hp, s[i].atk);
                    else
                    {
                        if (s[i].camp == 0 && s[i].location == n + 1){
                            printf("%s %s %d reached blue headquarter with %d elements and force %d\n", Camp[s[i].camp], name[s[i].type], s[i].id, s[i].hp, s[i].atk);
                            s[i].arrive = 1;
                            Lose[1]++;
                            if(Lose[1]==2){
                                Time(nowH, nowT);
                                printf("blue headquarter was taken\n");
                            }
                        }
                        if (s[i].camp == 1 && s[i].location == 0){
                            printf("%s %s %d reached red headquarter with %d elements and force %d\n", Camp[s[i].camp], name[s[i].type], s[i].id, s[i].hp, s[i].atk);
                            s[i].arrive = 1;
                            Lose[0]++;
                            if(Lose[0]==2){
                                Time(nowH, nowT);
                                printf("red headquarter was taken\n");
                            }
                        }
                    }
                }
        }
        if(nowT==20){for(int i=1;i<=n;i++)city[i]+=10;}
        if(nowT==30){
            sort(s+1,s+peo+1);
            for(int i=1;i<=n;i++)numcity[i]=0;
            for(int i=1;i<=peo;i++)if(!s[i].arrive&&!s[i].dead&&s[i].hp>0)numcity[s[i].location]++;
            for(int i=1;i<=peo;i++)
                if(!s[i].dead&&!s[i].arrive&&numcity[s[i].location]==1){
                    int c=s[i].location;
                    Time(nowH,nowT);
                    printf("%s %s %d earned %d elements for his headquarter\n",Camp[s[i].camp],name[s[i].type],s[i].id,city[c]);
                    Hp[s[i].camp]+=city[c];city[c]=0;
                }
        }
        if (nowT == 35){
            sort(s+1,s+peo+1);
            for(int i=1;i<=peo;i++){
                if(!s[i].arrive&&!s[i].dead&&s[i].camp==0&&have_arrow(s[i])){
                    int j=i+1;
                    while(j<=peo&&!s[j].dead){
                        if(s[j].location==s[i].location+1&&s[j].camp!=s[i].camp)break;
                        if(s[j].location>s[i].location+1)break;
                        j++;
                    }
                    if(j<=peo&&!s[j].dead&&s[j].location==s[i].location+1&&s[j].camp!=s[i].camp){
                        s[j].hp-=R;use_arrow(s[i]);
                        Time(nowH,nowT);
                        printf("red %s %d shot",name[s[i].type],s[i].id);
                        if(s[j].hp<=0)
                            printf(" and killed blue %s %d",name[s[j].type],s[j].id);
                        puts("");
                    }
                }
                if(!s[i].arrive&&!s[i].dead&&s[i].camp==1&&have_arrow(s[i])){
                    int j=i-1;
                    while(j&&!s[j].dead){
                        if(s[j].location==s[i].location-1&&s[j].camp!=s[i].camp)break;
                        if(s[j].location<s[i].location-1)break;
                        j--;
                    }
                    if(j&&!s[j].dead&&s[j].location==s[i].location-1&&s[j].camp!=s[i].camp){
                        s[j].hp-=R;use_arrow(s[i]);
                        Time(nowH,nowT);
                        printf("blue %s %d shot",name[s[i].type],s[i].id);
                        if(s[j].hp<=0)
                            printf(" and killed red %s %d",name[s[j].type],s[j].id);
                        puts("");
                    }
                }
            }
            for(int i=1;i<peo;i++)
                if(s[i].hp>0&&!s[i].arrive)
                    if(s[i+1].hp>0 && s[i].location==s[i+1].location){
                        node s1=s[i],s2=s[i+1];int c=s[i].location;
                        if(have_bomb(s[i])&&attack(s1,s2,(flag[c]==sgn(s[i].camp))||(!flag[c]&&(c&1)))){
                            Time(nowH,nowT+3);
                            printf("%s %s %d used a bomb and killed %s %s %d\n",Camp[s1.camp],name[s1.type],s1.id,Camp[s2.camp],name[s2.type],s2.id);
                            s[i+1].dead=s[i].dead=1;s[i].hp=s[i+1].hp=-1;
  //                          win[c][0]=win[c][1]=0;
                        }
                        if(have_bomb(s[i+1])&&attack(s2,s1,!((flag[c]==sgn(s[i].camp))||(!flag[c]&&(c&1))))){
                            Time(nowH,nowT+3);
                            printf("%s %s %d used a bomb and killed %s %s %d\n",Camp[s2.camp],name[s2.type],s2.id,Camp[s1.camp],name[s1.type],s1.id);
                            s[i].dead=s[i+1].dead=1;s[i].hp=s[i+1].hp=-1;
//                            win[c][0]=win[c][1]=0;
                        }
                    }
        }
        if (nowT == 40){
            sort(s + 1, s + peo + 1);
            for(int i=1;i<=peo;i++)s[i].ID=i;
            aw[0]=aw[1]=0;awl[0].clear();awl[1].clear();
            for (int i = 1; i < peo; i++)
                if (!s[i].dead && !s[i].arrive && !s[i + 1].dead && !s[i + 1].arrive)
                    if (s[i].location == s[i + 1].location){
                        int c = s[i].location,pre1=s[i].hp,pre2=s[i+1].hp;
                        if(!check(s[i],s[i+1],c))goto TRY;
                        real_attack(s[i],s[i+1]);
                        //printf("%d %d\n",win[c][0],win[c][1]);
                        if(s[i].hp<=0){
                            Time(nowH,nowT);printf("%s %s %d was killed in city %d\n",Camp[s[i].camp],name[s[i].type],s[i].id,c);
                            s[i].dead=1;
                        }
                        if(s[i+1].hp<=0){
                            Time(nowH,nowT);printf("%s %s %d was killed in city %d\n",Camp[s[i+1].camp],name[s[i+1].type],s[i+1].id,c);
                            s[i+1].dead=1;
                        }
                        if(check(s[i],s[i+1],c)){
                            if(s[i].type==0){
                                s[i].morale-=0.2;
                                if(s[i].morale>=0.7&&(flag[c]==sgn(s[i].camp)||(!flag[c]&&(c&1)))){
                                    Time(nowH,nowT);
                                    printf("%s %s %d yelled in city %d\n",Camp[s[i].camp],name[s[i].type],s[i].id,c);
                                }
                            }
                            if(s[i+1].type==0){
                                s[i+1].morale-=0.2;
                                if(s[i+1].morale>=0.7&&(flag[c]==sgn(s[i+1].camp)||(!flag[c]&&(c%2==0)))){
                                    Time(nowH,nowT);
                                    printf("%s %s %d yelled in city %d\n",Camp[s[i+1].camp],name[s[i+1].type],s[i+1].id,c);
                                }
                            }
                            if(s[i].type==3)s[i].loyalty-=k;
                            if(s[i+1].type==3)s[i+1].loyalty-=k;
                            win[c][0]=win[c][1]=0;
                        }
                        else{
                            if(s[i].hp>0&&s[i+1].type==3)s[i].hp+=pre2;
                            if(s[i+1].hp>0&&s[i].type==3)s[i+1].hp+=pre1;
                        }
                        TRY:
                        //printf("%d:%d %d\n",c,win[c][0],win[c][1]);
                        if(win[c][0]>=2&&flag[c]!=1){
                            flag[c]=1;
                            Time(nowH,nowT);printf("red flag raised in city %d\n",c);
                        }
                        else if(win[c][1]>=2&&flag[c]!=-1){
                            flag[c]=-1;
                            Time(nowH,nowT);printf("blue flag raised in city %d\n",c);
                        }
                    }
            sort(awl[0].begin(),awl[0].end(),cmp1);
            sort(awl[1].begin(),awl[1].end(),cmp2);
            for(int i=0;i<awl[0].size();i++)
                if(Hp[0]>=8){Hp[0]-=8;s[awl[0][i].second].hp+=8;}
            for(int i=0;i<awl[1].size();i++)
                if(Hp[1]>=8){Hp[1]-=8;s[awl[1][i].second].hp+=8;}
            Hp[0]+=aw[0];Hp[1]+=aw[1];
            for(int i=1;i<=peo;i++)
                if(s[i].hp<=0)s[i].dead=1;
        }
        if (nowT == 50){
            Time(nowH, nowT);printf("%d elements in red headquarter\n", Hp[0]);
            Time(nowH, nowT);printf("%d elements in blue headquarter\n", Hp[1]);
        }
        if (nowT == 55){
            sort(s + 1, s + peo + 1 , CMP);
            for (int i = 1; i <= peo; i++)
                if (!s[i].dead){
                    sort(s[i].weapon.begin(), s[i].weapon.end());
                    num[0] = num[1] = num[2] = 0;
                    int _used=0,ATK=0;
                    for (auto tmp : s[i].weapon){
                        num[tmp.type]++;
                        if(tmp.type==0)ATK=tmp.atk;
                        if(tmp.type==2)_used=tmp.used;
                    }
                    Time(nowH, nowT);
                    printf("%s %s %d has ", Camp[s[i].camp], name[s[i].type], s[i].id);
                    int FLAG=0;
                    if(num[0]+num[1]+num[2]==0)printf("no weapon\n");
                    else{
                        if(num[2]){
                            FLAG=1;
                            printf("arrow(%d)",3-_used);
                        }
                        if(num[1]){
                            if(!FLAG)printf("bomb");
                            else printf(",bomb");
                            FLAG=1;
                        }
                        if(num[0]){
                            if(!FLAG)printf("sword(%d)",ATK);
                            else printf(",sword(%d)",ATK);
                            FLAG=1;
                        }
                        puts("");
                    }

                }
        }
        nowT += 5; T -= 5;
        if (nowT == 60){nowH++;nowT = 0;}
    }
    peo = 0; Numb = Numr = 0;
    for(int i=1;i<=n;i++)city[i]=0,win[i][0]=win[i][1]=0,flag[i]=0;
}
int main()
{
#ifndef ONLINE_JUDGE
    freopen("in.in", "r", stdin);
    freopen("out.out", "w", stdout);
#endif
    int t;
    scanf("%d", &t);
    for (int i = 1; i <= t; i++)
    {
        printf("Case %d:\n", i);
        work();
    }
    return 0;
}
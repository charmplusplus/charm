#
# Example spec file for cdplayer app...
#
Summary: Charm++
Name: Charm
Version: 5.6
Release: 10
Copyright: GPL
Group: Applications/System
BuildRoot: /var/tmp/%{name}-root
Source: http://charm.cs.uiuc.edu/distrib/Charm-5.6.tar.bz2
URL: http://charm.cs.uiuc.edu
Vendor: PPL <ppl@uiuc.edu>

%description
Charm++ for Redhat

%prep
%setup -n %{name}-%{version}
bzip2 -dc  %{_sourcedir}/%{name}-%{version}.tar.bz2 | tar xvf -
%build
./build AMPI net-linux -O -DCMK_OPTIMIZE
%clean
rm -rf $RPM_BUILD_ROOT
%install
mkdir -p $RPM_BUILD_ROOT
cd tmp; make DESTDIR=$RPM_BUILD_ROOT/usr/local/charm  install 
%post
 
%files
/usr/local/charm/*
%doc README LICENSE CHANGES README.cygwin README.win32
